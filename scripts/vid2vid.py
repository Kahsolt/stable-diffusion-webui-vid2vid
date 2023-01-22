import os
import sys
import json
import html
import shutil
from pathlib import Path
from subprocess import Popen
from PIL import Image
from PIL.Image import Image as PILImage
from enum import Enum
from collections import Counter
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Callable, Union, Any
import pickle as pkl
from traceback import print_exc, format_exc
from time import time
import gc

import gradio as gr
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import Tensor 
import numpy as np
import cv2

from modules.scripts import Script
from modules.script_callbacks import remove_callbacks_for_function, on_before_image_saved, ImageSaveParams, on_cfg_denoiser, CFGDenoiserParams
from modules.devices import torch_gc, autocast, device, cpu
from modules.shared import opts, state
from modules.processing import Processed, StableDiffusionProcessingImg2Img, get_fixed_seed

try:
    # should be <sd-webui> root abspath
    SD_WEBUI_PATH = Path.cwd()
    # prompt-travel
    PTRAVEL_PATH = Path(SD_WEBUI_PATH) / 'extensions' / 'stable-diffusion-webui-prompt-travel'
    assert PTRAVEL_PATH.exists() ; sys.path.insert(0, str(PTRAVEL_PATH))
    from scripts.prompt_travel import process_images_before, process_images_after, process_images_prompt_to_cond, process_images_cond_to_image
    from scripts.prompt_travel import cond_replace
    from scripts.prompt_travel import weighted_sum, geometric_slerp
    # bundled tools
    TOOL_PATH   = Path(PTRAVEL_PATH) / 'tools'
    RESR_PATH   = Path(TOOL_PATH) / 'realesrgan-ncnn-vulkan'
    RESR_BIN    = Path(RESR_PATH) / 'realesrgan-ncnn-vulkan.exe'
    RIFE_PATH   = Path(TOOL_PATH) / 'rife-ncnn-vulkan'
    RIFE_BIN    = Path(RIFE_PATH) / 'rife-ncnn-vulkan.exe'
    FFPROBE_BIN = Path(TOOL_PATH) / 'ffmpeg' / 'bin' / 'ffprobe.exe'
    FFMPEG_BIN  = Path(TOOL_PATH) / 'ffmpeg' / 'bin' / 'ffmpeg.exe'
    assert Path(TOOL_PATH)  .exists()
    assert Path(RESR_BIN)   .exists()
    assert Path(RIFE_BIN)   .exists()
    assert Path(FFPROBE_BIN).exists()
    assert Path(FFMPEG_BIN) .exists()
    # deepdanbooru
    from modules.deepbooru import model as deepbooru_model
    # midas
    MIDAS_REPO_PATH  = Path(SD_WEBUI_PATH) / 'repositories' / 'midas'
    MIDAS_MODEL_FILE = Path(SD_WEBUI_PATH) / 'models' / 'midas' / 'midas_v21_small-70d6b9c8.pt'
    MIDAS_MODEL_HW   = 256
    assert MIDAS_REPO_PATH .exists()
    assert MIDAS_MODEL_FILE.exists()
    from torchvision.transforms import Compose
    from repositories.midas.midas.midas_net_custom import MidasNet_small
    from repositories.midas.midas.transforms import Resize, NormalizeImage, PrepareForNet
except:
    print_exc()
    raise RuntimeError('<< integrity check failed, please check your installation :(')

class ImageFormat(Enum):
    JPG  = 'jpg'
    PNG  = 'png'
    WEBP = 'webp'

class VideoFormat(Enum):
    MP4  = 'mp4'
    GIF  = 'gif'
    WEBM = 'webm'
    AVI  = 'avi'

def get_resr_model_names() -> List[str]:
    fbases = {fn.stem for fn in (RESR_PATH / 'models').iterdir()}
    return list({n[:-3] if n[-3:] in ['-x2', '-x3', '-x4'] else n for n in fbases})

def get_rife_model_names() -> List[str]:
    return [fn.name for fn in RIFE_PATH.iterdir() if fn.is_dir()]

if 'global consts':
    # cache folder layout
    WS_CONFIG               = 'config.json'
    WS_FFPROBE              = 'ffprobe.json'
    WS_FRAMES               = 'frames'
    WS_AUDIO                = 'audio.wav'
    WS_MASK                 = 'depthmask'
    WS_TAGS                 = 'tags.json'
    WS_LATENT               = 'latent'
    WS_ICOND                = 'icond'
    WS_TRAVEL               = 'ptravel'
    WS_RESR                 = 'resr'
    WS_RIFE                 = 'rife'
    WS_SYNTH                = 'synth'    # stem

    WS_TAGS_TOPK            = 'tags-topk.txt'
    WS_FRAMES_DIST          = 'frames_dist.npy'
    WS_TRAVEL_STEPS         = 'travel_steps.png'
    WS_BUG_DUMP             = 'bug_dump.npy'

    __ = lambda key, value=None: opts.data.get(f'customscript/vid2vid.py/img2img/{key}/value', value)

    LABEL_CACHE_FOLDER      = 'Cache Folder'
    LABEL_WORKSPACE_FOLDER  = 'Workspace Folder'
    LABEL_VIDEO_FILE        = 'Input video file'
    LABEL_VIDEO_INFO        = 'Video media info'
    LABEL_EXTRACT_FMT       = 'Extracted file format'
    LABEL_EXTRACT_FPS       = 'Extracted FPS'
    LABEL_IC_PROMPT         = 'InvCond init prompt'
    LABEL_IC_STEP           = 'InvCond steps'
    LABEL_IC_GRAD_ACC       = 'InvCond grad accumulate ratio'
    LABEL_IC_LR             = 'InvCond learning rate'
    LABEL_TRAVEL_RATIO      = 'Travel interp ratio'
    LABEL_TRAVEL_DAMP       = 'Travel damping'
    LABEL_INIT_NOISE_W      = 'Init noise weight'
    LABEL_USE_MASK          = 'Use depth as mask'
    LABEL_MASK_LOWCUT       = 'Mask low-cut'
    LABEL_RESR_MODEL        = 'Real-ESRGAN model'
    LABEL_RESR_RATIO        = 'Upscale ratio'
    LABEL_RIFE_MODEL        = 'RIFE model'
    LABEL_RIFE_FPS          = 'Interpolated FPS for export'
    LABEL_EXPORT_FMT        = 'Export format'
    LABEL_COMPOSE_SRC       = 'Frame source'
    LABEL_ALLOW_OVERWRITE   = 'Allow overwrite cache'

    CHOICES_IMAGE_FMT       = [x.value for x in ImageFormat]
    CHOICES_VIDEO_FMT       = [x.value for x in VideoFormat]
    CHOICES_RESR_MODEL      = get_resr_model_names()
    CHOICES_RESR_RATIO      = [2, 3, 4, 6, 8, 12, 16]
    CHOICES_RIFE_MODEL      = get_rife_model_names()
    CHOICES_COMPOSE_SRC     = [
        WS_FRAMES,
        WS_MASK,
        WS_TRAVEL,
        WS_RESR,
        WS_RIFE,
    ]

    INIT_CACHE_FOLDER = Path(os.environ['TMP']) / 'sd-webui-vid2vid'
    INIT_CACHE_FOLDER.mkdir(exist_ok=True)

    DEFAULT_CACHE_FOLDER    = __(LABEL_CACHE_FOLDER, str(INIT_CACHE_FOLDER))
    DEFAULT_EXTRACT_FMT     = __(LABEL_EXTRACT_FMT, ImageFormat.JPG.value)
    DEFAULT_EXTRACT_FPS     = __(LABEL_EXTRACT_FPS, 8)
    DEFAULT_IC_PROMPT       = __(LABEL_IC_PROMPT, '(masterpiece:1.3), highres, ')
    DEFAULT_IC_STEP         = __(LABEL_IC_STEP, 32)
    DEFAULT_IC_GRAD_ACC     = __(LABEL_IC_GRAD_ACC, 4)
    DEFAULT_IC_LR           = __(LABEL_IC_LR, '0.001')
    DEFAULT_TRAVEL_RATIO    = __(LABEL_TRAVEL_RATIO, 4)
    DEFAULT_TRAVEL_DAMP     = __(LABEL_TRAVEL_DAMP, 1.0)
    DEFAULT_INIT_NOISE_W    = __(LABEL_INIT_NOISE_W, 0.95)
    DEFAULT_USE_MASK        = __(LABEL_USE_MASK, False)
    DEFAULT_MASK_LOWCUT     = __(LABEL_MASK_LOWCUT, 16)
    DEFAULT_RESR_MODEL      = __(LABEL_RESR_MODEL, 'realesr-animevideov3')
    DEFAULT_RESR_RATIO      = __(LABEL_RESR_RATIO, 2)
    DEFAULT_RIFE_MODEL      = __(LABEL_RIFE_MODEL, 'rife-v4')
    DEFAULT_RIFE_FPS        = __(LABEL_RIFE_FPS, 30)
    DEFAULT_COMPOSE_SRC     = __(LABEL_COMPOSE_SRC, WS_RIFE)
    DEFAULT_EXPORT_FMT      = __(LABEL_EXPORT_FMT, VideoFormat.MP4.value)
    DEFAULT_ALLOW_OVERWRITE = __(LABEL_ALLOW_OVERWRITE, True)

    INVCOND_HELP_HTML = '''
<div>
  <h4> Instructions for InvCond: 🔤 </h4>
  <p> 1. basic flow is like training embeddings (textual-inversion), yes it is very slow :( </p>
  <p> 2. fill the init prompt with cherry-picked inverted tags, describing your video content 
      and also adding something new </p> 
  <p> 3. the 'grad_acc_ratio' is a multiplier to 'steps', it averages losses to make training stable </p>
</div>
'''

    PTRAVEL_HELP_HTML = '''
<div>
  <h4> Instructions for this step: 😉 </h4>
  <p> 1. check settings below, and also <strong>all img2img settings</strong> top above ↑↑,
      remeber to <strong>select a dummy init image</strong> to avoid webui "AttributeError" error :) </p>
  <p> 2. load <strong>hypernetworks</strong> and <strong>embeddings</strong> if you want
  <p> 3. the specified "prompt" will be <strong>prepended</strong> to each inverted tags-string, 
      while the "negative_prompt" will be used for all </p>
  <p> 4. click the top-right main <strong>"Generate"</strong> button to start! </p>
</div>
'''


def get_folder_file_count(dp:Union[Path, str]) -> int:
    return len(os.listdir(dp))

def get_file_size(fp:Union[Path, str]) -> float:
    return os.path.getsize(fp) / 2**20


@dataclass
class Config:
    # Step 1: Select video file, probe media info, extract frames
    workspace: str = None
    extract_fps = DEFAULT_EXTRACT_FPS
    extract_fmt = DEFAULT_EXTRACT_FMT

    # Step 2: Make depth mask and inverted tags

    # Step 3: Prompt travel
    travel_ratio = DEFAULT_TRAVEL_RATIO
    travel_damp  = DEFAULT_TRAVEL_DAMP
    use_mask     = DEFAULT_USE_MASK
    mask_lowcut  = DEFAULT_MASK_LOWCUT
    # inherit all other settings in img2img control panel :)
    i2i_negative_prompt    = None
    i2i_resize_mode        = None
    i2i_sampler_name       = None
    i2i_steps              = None
    i2i_width              = None
    i2i_height             = None
    i2i_cfg_scale          = None
    i2i_denoising_strength = None
    i2i_seed               = None
    i2i_subseed            = None
    i2i_subseed_strength   = None

    # Step 4: Image super-resolution & Frame interpolation
    resr_model = DEFAULT_RESR_MODEL
    resr_ratio = DEFAULT_RESR_RATIO
    rife_model = DEFAULT_RIFE_MODEL
    rife_fps   = DEFAULT_RIFE_FPS

    # Step 5: Render
    export_fmt = DEFAULT_EXPORT_FMT

    @classmethod
    def load(cls):
        global cur_config

        fp = Path(workspace) / 'config.json'
        if fp.exists():
            with open(fp, 'r', encoding='utf-8') as fh:
                cur_config = cls(**json.load(fh))
        else:
            cur_config = cls()

    def save(self, close=False):
        global cur_config

        fp = Path(workspace) / 'config.json'
        with open(fp, 'w', encoding='utf-8') as fh:
            json.dump(asdict(self), fh, indent=2, ensure_ascii=False)
        if close: cur_config = None

# global runtime vars
workspace: Path = None
cur_config: Config = None
cur_cache_folder: Path = Path(DEFAULT_CACHE_FOLDER)
cur_allow_overwrite: bool = DEFAULT_ALLOW_OVERWRITE
cur_task: str = None

GradioRequest = Dict[str, Any]

class RetCode(Enum):
    INFO  = 'info'
    WARN  = 'warn'
    ERROR = 'error'

def gr_update_status(text=None, code=RetCode.INFO, task:str=None, ts:float=None) -> GradioRequest:
    if not text: return gr.HTML.update()
    safe_text = html.escape(text)
    task_str = f' {task!r}' if task else ''
    ts_str = f' ({ts:.3f}s)' if ts else ''
    TEMPLATES = {
        RetCode.INFO:  lambda: gr.HTML.update(value=f'<div style="color:blue">Done{task_str}!{ts_str} </br> => {safe_text}</div>'),
        RetCode.WARN:  lambda: gr.HTML.update(value=f'<div style="color:green">Warn{task_str}! </br> => {safe_text}</div>'),
        RetCode.ERROR: lambda: gr.HTML.update(value=f'<div style="color:red">Error{task_str}! </br> => {safe_text}</div>'),
    }
    return TEMPLATES[code]()

def get_workspace_path(cache_folder:str, fn:str) -> Path:
    def safe_for_path(fn:str) -> str:
        name = fn.replace(' ', '_')
        name = name[:32]    # make things short
        return name

    return Path(cache_folder) / safe_for_path(fn)

def _file_select(video_file:object) -> List[GradioRequest]:
    global workspace, cur_cache_folder

    # close workspace
    if video_file is None:
        cur_config.save(close=True)
        ws_name = workspace.name
        workspace = None

        return [
            gr.Text.update(label=LABEL_CACHE_FOLDER, value=cur_cache_folder, interactive=True),
            gr.TextArea.update(visible=False),
            gr_update_status(f'closed workspace {ws_name!r}'),
        ]
    
    # open existing workspace
    ws_dp = get_workspace_path(cur_cache_folder, video_file.orig_name)
    info_fp = Path(ws_dp) / WS_FFPROBE
    if ws_dp.exists():
        workspace = ws_dp
        ws_name = workspace.name
        Config.load()
        cur_config.workspace = str(workspace)
        cur_config.save()

        with open(info_fp, 'r', encoding='utf-8') as fh:
            media_info = json.dumps(json.load(fh), indent=2, ensure_ascii=False)
        
        return [
            gr.Text.update(label=LABEL_WORKSPACE_FOLDER, value=workspace, interactive=False),
            gr.TextArea.update(value=media_info, visible=True),
            gr_update_status(f'open workspace {ws_name!r}'),
        ]
    
    # try create new workspace
    cmd = f'"{FFPROBE_BIN}" -i "{video_file.name}" -show_streams -of json'
    print(f'>> exec: {cmd}')
    try:
        media_info = json.dumps(json.loads(os.popen(cmd).read().strip()), indent=2, ensure_ascii=False)
        
        ws_dp.mkdir(parents=True)
        workspace = ws_dp
        ws_name = workspace.name
        Config.load()

        with open(info_fp, 'w', encoding='utf-8') as fh:
            fh.write(media_info)

        return [
            gr.Text.update(label=LABEL_WORKSPACE_FOLDER, value=workspace, interactive=False),
            gr.TextArea.update(value=media_info, visible=True),
            gr_update_status(f'create new workspace {ws_name!r}'),
        ]
    except:
        e = format_exc() ; print(e)
        return [
            gr.Text.update(),
            gr.TextArea.update(visible=False),
            gr_update_status(e, code=RetCode.ERROR),
        ]

def _txt_working_folder(working_folder:str) -> GradioRequest:
    global workspace, cur_cache_folder

    # Mode: workspace folder
    if workspace is not None: return gr_update_status()

    # Mode: cache folder
    if Path(working_folder).is_dir():
        cur_cache_folder = working_folder
        return gr_update_status(f'set cache folder path: {cur_cache_folder}')
    else:
        return gr_update_status(f'invalid folder path: {working_folder}', code=RetCode.WARN)

def _btn_open(working_folder:str) -> GradioRequest:
    if Path(working_folder).is_dir():
        os.startfile(working_folder)
        return gr_update_status(f'open folder: {working_folder!r}')
    else:
        return gr_update_status(f'invalid folder path: {working_folder!r}', code=RetCode.ERROR)

def _chk_overwrite(allow_overwrite:bool) -> None:
    global cur_allow_overwrite

    cur_allow_overwrite = allow_overwrite


TaskResponse = Tuple[RetCode, str]

def task(fn:Callable[..., TaskResponse]):
    def wrapper(*args, **kwargs):
        global cur_task
        if workspace is None: 
            code = RetCode.ERROR
            info = 'no current workspace opened!'
            ts = None
        elif cur_task is not None: 
            code = RetCode.ERROR
            info = f'task {cur_task!r} is stilling running!'
            ts = None
        else:
            cur_task = fn.__name__
            _ts = time()
            code, info = fn(*args, **kwargs)
            ts = time() - _ts
            cur_task = None
        return gr_update_status(info, code=code, task=fn.__name__, ts=ts)
    return wrapper

def task_ignore_str(taskname: str) ->str:
    return f'task "{taskname}" ignored due to cached already exists :)'

@task
def _btn_ffmpeg_extract(video_file:object, extract_fmt:str, extract_fps:float) -> TaskResponse:
    out_dp = workspace / WS_FRAMES
    if out_dp.exists():
        if not cur_allow_overwrite:
            return RetCode.WARN, task_ignore_str('extract')
        shutil.rmtree(str(out_dp))
        dist_fp = workspace / WS_FRAMES_DIST
        if dist_fp.exists():
            dist_fp.unlink()
    out_dp.mkdir()

    out_fp = workspace / WS_AUDIO
    if out_fp.exists(): out_fp.unlink()

    try:
        cmd = f'"{FFMPEG_BIN}" -i "{video_file.name}" -an -r {extract_fps} -f image2 -q:v 2 {out_dp}\\%08d.{extract_fmt}'
        print(f'>> exec: {cmd}')
        Popen(cmd, shell=True, text=True, encoding='utf-8').wait()
        try:
            cmd = f'"{FFMPEG_BIN}" -i "{video_file.name}" -vn "{out_fp}"'
            Popen(cmd, shell=True, text=True, encoding='utf-8').wait()
            has_audio = 'yes'
        except:
            has_audio = 'no'
            print_exc()
        
        cur_config.workspace = str(workspace)
        cur_config.extract_fmt = extract_fmt
        cur_config.extract_fps = extract_fps
        cur_config.save()

        return RetCode.INFO, f'frames: {get_folder_file_count(out_dp)}, audio: {has_audio}'
    except:
        e = format_exc() ; print(e)
        return RetCode.ERROR, e

@task
def _btn_latent() -> TaskResponse:
    in_dp = workspace / WS_FRAMES
    if not in_dp.exists():
        return RetCode.ERROR, f'frames folder not found: {in_dp}'

    out_dp = workspace / WS_LATENT
    if out_dp.exists():
        if not cur_allow_overwrite:
            return RetCode.WARN, task_ignore_str('latent')
        shutil.rmtree(str(out_dp))
    out_dp.mkdir()

    from modules.shared import sd_model

    try:
        with torch.no_grad(), autocast():
            for fp in tqdm(list(in_dp.iterdir())):
                img = Image.open(fp).convert('RGB')
                im = np.array(img, dtype=np.uint8)
                im = (im / 127.5 - 1.0).astype(np.float32)
                X = torch.from_numpy(im).permute(2, 0, 1).to(device).unsqueeze(dim=0)

                x:Tensor = sd_model.get_first_stage_encoding(sd_model.encode_first_stage(X))   # [B=1, C=4, H, W]

                x.requires_grad = False
                torch.save(x.detach().cpu(), out_dp / fp.with_suffix('.t').name)

        return RetCode.INFO, f'latents: {get_folder_file_count(out_dp)}'
    except Exception:
        e = format_exc() ; print(e)
        return RetCode.ERROR, e
    finally:
        torch_gc()
        gc.collect()

@task
def _btn_midas() -> TaskResponse:
    in_dp = workspace / WS_FRAMES
    if not in_dp.exists():
        return RetCode.ERROR, f'frames folder not found: {in_dp}'

    out_dp = workspace / WS_MASK
    if out_dp.exists():
        if not cur_allow_overwrite:
            return RetCode.WARN, task_ignore_str('midas')
        shutil.rmtree(str(out_dp))
    out_dp.mkdir()

    try:
        cuda = torch.device('cuda')
        model = MidasNet_small(MIDAS_MODEL_FILE).to(device)
        if device == cuda: model = model.to(memory_format=torch.channels_last)  
        model.eval()

        transform = Compose([
            Resize(
                MIDAS_MODEL_HW,
                MIDAS_MODEL_HW,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method='upper_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        with torch.no_grad(), autocast():
            for fn in tqdm(list(in_dp.iterdir())):                  # TODO: make batch for speedup
                if state.interrupted: break

                img = Image.open(fn).convert('RGB')
                im = np.asarray(img, dtype=np.float32) / 255.0      # [H, W, C], float32

                X_np = transform({'image': im})['image']            # [C, maxH, maxW], float32
                X = torch.from_numpy(X_np).to(device).unsqueeze(0)  # [B=1, C, maxH, maxW], float32
                if device == cuda: X = X.to(memory_format=torch.channels_last)  

                pred = model.forward(X)

                depth = F.interpolate(pred.unsqueeze(1), size=im.shape[:2], mode='bicubic', align_corners=False).squeeze().cpu().numpy()   # [H, W], float16
                vmin, vmax = depth.min(), depth.max()
                if vmax - vmin > np.finfo(depth.dtype).eps:
                    depth_n = (depth - vmin) / (vmax - vmin)
                else:
                    depth_n = np.zeros_like(depth)

                depth_u8 = (depth_n * np.iinfo(np.uint8).max).astype(np.uint8)
                img_out = Image.fromarray(depth_u8)
                img_out.save(Path(out_dp) / f'{Path(fn).stem}.png')

        return RetCode.INFO, f'frames: {get_folder_file_count(out_dp)}'
    except Exception:
        e = format_exc() ; print(e)
        return RetCode.ERROR, e
    finally:
        if model in locals(): del model
        torch_gc()
        gc.collect()

@task
def _btn_deepdanbooru(topk=32) -> TaskResponse:
    in_dp = workspace / WS_FRAMES
    if not in_dp.exists():
        return RetCode.ERROR, f'frames folder not found: {in_dp}'

    out_fp = workspace / WS_TAGS
    if out_fp.exists():
        if not cur_allow_overwrite:
            return RetCode.WARN, task_ignore_str('deepdanbooru')
        out_fp.unlink()

    try:
        tags: Dict[str, str] = { }
        deepbooru_model.start()
        for fp in tqdm(list(in_dp.iterdir())):
            img = Image.open(fp).convert('RGB')
            tags[fp.name] = deepbooru_model.tag_multi(img)

        tags_flatten = []
        for prompt in tags.values():
            tags_flatten.extend([t.strip() for t in prompt.split(',')])
        tags_topk = ', '.join([t for t, c in Counter(tags_flatten).most_common(topk)])

        with open(out_fp, 'w', encoding='utf-8') as fh:
            json.dump(tags, fh, indent=2, ensure_ascii=False)
        with open(workspace / WS_TAGS_TOPK, 'w', encoding='utf-8') as fh:
            fh.write(tags_topk)

        return RetCode.INFO, f'images: {len(tags)}, top-{topk} freq tags: {tags_topk}'
    except:
        e = format_exc() ; print(e)
        return RetCode.ERROR, e
    finally:
        deepbooru_model.model.to(cpu)
        torch_gc()
        gc.collect()

@task
def _btn_invert_cond(ic_prompt:str, ic_step:int, ic_grad_acc:int, ic_lr:str) -> TaskResponse:
    in_dp = workspace / WS_FRAMES
    if not in_dp.exists():
        return RetCode.ERROR, f'frames folder not found: {in_dp}'

    latent_dp = workspace / WS_LATENT
    if not latent_dp.exists():
        return RetCode.ERROR, f'latent folder not found: {latent_dp}'

    out_dp = workspace / WS_ICOND
    if out_dp.exists():
        if not cur_allow_overwrite:
            return RetCode.WARN, task_ignore_str('invert_cond')
        shutil.rmtree(str(out_dp))
    out_dp.mkdir()

    from torch.optim import AdamW, SGD
    import modules.devices as devices
    from modules.textual_inversion.textual_inversion import shared, sd_hijack_checkpoint, LearnRateScheduler
    import matplotlib.pyplot as plt
    from ldm.models.diffusion.ddpm import LatentDiffusion

    old_parallel_processing_allowed = shared.parallel_processing_allowed

    try:
        sd_model:LatentDiffusion = shared.sd_model
        if sd_model.model.conditioning_key in {'hybrid', 'concat'}:
            return RetCode.ERROR, 'inversion on an inpainting model is not supported yet :('

        if not 'low mem':
            sd_hijack_checkpoint.add()

        # kanban
        fps = list(in_dp.iterdir())
        n_job = len(fps)
        state.job_count = n_job
        state.interrupted = False
        
        # init cond
        with torch.no_grad(), autocast():
            init_c: Tensor = sd_model.cond_stage_model([ic_prompt]).to(device)    # [B=1, T=77, D=768]

        if 'unload':
            shared.parallel_processing_allowed = False
            shared.sd_model.first_stage_model.to(devices.cpu)
            shared.sd_model.cond_stage_model.to(devices.cpu)

        # invert for each image
        for i, fp in enumerate(tqdm(fps)):
            if state.interrupted: break
            state.job_no = i

            # latent
            x = torch.load(latent_dp / fp.with_suffix('.t').name).to(device)    # [B=1, T=77, D=768]

            if not 'show x':
                tmp = x.squeeze().detach().cpu().numpy()
                plt.clf()
                plt.subplot(221) ; plt.imshow(tmp[0])
                plt.subplot(222) ; plt.imshow(tmp[1])
                plt.subplot(223) ; plt.imshow(tmp[2])
                plt.subplot(224) ; plt.imshow(tmp[3])
                plt.suptitle('x')
                plt.show()
                del tmp

            # cond
            c = init_c.detach().clone()    # [B=1, T=77, D=768]
            c.requires_grad = True

            if not 'show c':
                tmp = c.squeeze().detach().cpu().numpy()
                plt.clf()
                plt.imshow(tmp)
                plt.suptitle('c')
                plt.show()
                del tmp

            scheduler = LearnRateScheduler(ic_lr, ic_step)
            #optimizer = AdamW([c], lr=scheduler.learn_rate, weight_decay=0)
            optimizer = SGD([c], lr=scheduler.learn_rate, momentum=0.9, weight_decay=0, nesterov=True)
            losses = []

            is_break_iter = False
            for j in range(ic_step):
                if scheduler.finished: break
                if state.interrupted: is_break_iter = True ; break

                scheduler.apply(optimizer, j)

                loss_acc = 0.0
                for _ in range(ic_grad_acc):
                    if state.interrupted: is_break_iter = True ; break

                    with autocast():
                        if 'fuzzy grad':
                            out = sd_model(x, c)
                        else:
                            # fix random resources, this seems not ok 🤔
                            t = torch.randint(0, sd_model.num_timesteps, (x.shape[0],), device=sd_model.device).long() * 0 + 512
                            noise = devices.randn(114514, x.shape)
                            out = sd_model.p_losses(x, c, t, noise=noise)

                        loss = out[0] / ic_grad_acc

                    loss_acc += loss.item()
                    loss.backward()         # calc grad

                if is_break_iter: break

                optimizer.step()
                optimizer.zero_grad()       # clear grad 
                losses.append(loss_acc)

                with torch.no_grad():
                    L1 = (c - init_c).abs()
                print(f'>> [frame {i+1}/{n_job}] (step {j+1}/{ic_step}) loss: {losses[-1]:.7f}, lr: {scheduler.learn_rate}, L1: {L1.mean()}, Linf: {L1.max()}')

            if is_break_iter: break

            # shift as a prior for the next (succesive)
            init_c = c
            c.requires_grad = False
            torch.save(c.detach().cpu(), out_dp / fp.with_suffix('.t').name)

            if not 'debug':
                debug_dp = workspace / 'debug'
                debug_dp.mkdir(exist_ok=True)

                plt.clf()
                plt.plot(losses)
                plt.suptitle('loss')
                plt.savefig(debug_dp / fp.with_suffix('.png').name, dpi=400)

            del x, c, losses, scheduler, optimizer
            gc.collect()

        return RetCode.INFO, f'iconds: {get_folder_file_count(out_dp)}'
    except:
        e = format_exc() ; print(e)
        return RetCode.ERROR, e
    finally:
        if 'inv unload':
            shared.sd_model.cond_stage_model.to(devices.device)
            shared.sd_model.first_stage_model.to(devices.device)
            shared.parallel_processing_allowed = old_parallel_processing_allowed
        if not 'low mem':
            sd_hijack_checkpoint.remove()
        torch_gc()
        gc.collect()

@task
def _btn_resr(resr_model:str, resr_ratio:int, extract_fmt:str) -> TaskResponse:
    in_dp = workspace / WS_TRAVEL
    if not in_dp.exists():
        return RetCode.ERROR, f'ptravel folder not found: {in_dp}'

    out_dp = workspace / WS_RESR
    if out_dp.exists():
        if not cur_allow_overwrite:
            return RetCode.WARN, task_ignore_str('resr')
        shutil.rmtree(str(out_dp))
    out_dp.mkdir()

    try:
        cmd = f'"{RESR_BIN}" -v -s {resr_ratio} -n {resr_model} -f {extract_fmt} -i "{in_dp}" -o "{out_dp}"'
        print(f'>> exec: {cmd}')
        Popen(cmd, shell=True, text=True, encoding='utf-8').wait()

        cur_config.resr_model = resr_model
        cur_config.resr_ratio = resr_ratio
        cur_config.save()

        return RetCode.INFO, f'frames: {get_folder_file_count(out_dp)}'
    except:
        e = format_exc() ; print(e)
        return RetCode.ERROR, e

@task
def _btn_rife(rife_model:str, rife_fps:float, extract_fmt:str, extract_fps:float, travel_ratio:int) -> TaskResponse:
    in_dp = workspace / WS_RESR
    if not in_dp.exists():
        return RetCode.ERROR, f'resr folder not found: {in_dp}'

    out_dp = workspace / WS_RIFE
    if out_dp.exists():
        if not cur_allow_overwrite:
            return RetCode.WARN, task_ignore_str('rife')
        shutil.rmtree(str(out_dp))
    out_dp.mkdir()

    try:
        n_interp = get_folder_file_count(in_dp) * rife_fps // (extract_fps * travel_ratio)
        cmd = f'"{RIFE_BIN}" -v -n {n_interp} -m {rife_model} -f %08d.{extract_fmt} -i "{in_dp}" -o "{out_dp}"'
        print(f'>> exec: {cmd}')
        Popen(cmd, shell=True, text=True, encoding='utf-8').wait()

        cur_config.rife_model = rife_model
        cur_config.rife_fps   = rife_fps
        cur_config.save()

        return RetCode.INFO, f'frames: {get_folder_file_count(out_dp)}'
    except:
        e = format_exc() ; print(e)
        return RetCode.ERROR, e

@task
def _btn_ffmpeg_compose(export_fmt:str, rife_fps:float, extract_fmt:str, src:str) -> TaskResponse:
    in_img = workspace / src
    if not in_img.exists():
        return RetCode.ERROR, f'src folder not found: {in_img}'

    opts = ''
    in_wav = workspace / WS_AUDIO
    if in_wav.exists():
        opts += f' -i "{in_wav}"'

    out_vid = workspace / f'{WS_SYNTH}-{src}.{export_fmt}'
    if out_vid.exists():
        if not cur_allow_overwrite:
            return RetCode.WARN, 'task "render" ignored due to cache exists'
        out_vid.unlink()

    fn_stem = r'%05d' if src == WS_TRAVEL else r'%08d'
    fn_ext = {
        WS_FRAMES: extract_fmt,
        WS_MASK:   'png',
        WS_TRAVEL: 'png',
        WS_RIFE:   extract_fmt,
        WS_RESR:   extract_fmt,
    }.get(src)

    try:
        cmd = f'"{FFMPEG_BIN}"{opts} -framerate {rife_fps} -i "{in_img}\\{fn_stem}.{fn_ext}" -crf 20 -c:v libx264 -pix_fmt yuv420p "{out_vid}"'
        print(f'>> exec: {cmd}')
        Popen(cmd, shell=True, text=True, encoding='utf-8').wait()
        
        cur_config.export_fmt = export_fmt
        cur_config.save()

        return RetCode.INFO, f'filesize: {get_file_size(out_vid):.3f}'
    except:
        e = format_exc() ; print(e)
        return RetCode.ERROR, e


class Script(Script):

    def title(self):
        return 'vid2vid'

    def describe(self):
        return 'Convert a video to an AI generated stuff.'

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        with gr.Blocks():
            with gr.Tab('1: Extract frames'):
                status_info_1 = gr.HTML()

                with gr.Row(variant='compact'):
                    working_folder = gr.Text(label=LABEL_CACHE_FOLDER, value=lambda: DEFAULT_CACHE_FOLDER, max_lines=1)
                    working_folder.change(fn=_txt_working_folder, inputs=working_folder, outputs=status_info_1, show_progress=False)
                    btn_open = gr.Button(value='\U0001f4c2', variant='tool')   # 📂
                    btn_open.click(fn=_btn_open, inputs=working_folder, outputs=status_info_1, show_progress=False)
                with gr.Row(variant='compact'):
                    video_file = gr.File(label=LABEL_VIDEO_FILE, file_types=['video'])
                    video_info = gr.TextArea(label=LABEL_VIDEO_INFO, max_lines=7, visible=False)
                    video_file.change(fn=_file_select, inputs=video_file, outputs=[working_folder, video_info, status_info_1], show_progress=False)

                with gr.Row(variant='compact').style(equal_height=True):
                    extract_fmt = gr.Radio(label=LABEL_EXTRACT_FMT, value=lambda: DEFAULT_EXTRACT_FMT, choices=CHOICES_IMAGE_FMT)
                    extract_fps = gr.Slider(label=LABEL_EXTRACT_FPS, value=lambda: DEFAULT_EXTRACT_FPS, minimum=1, maximum=24, step=0.1)
                    
                    btn_ffmpeg_extract = gr.Button('Extract frames!')
                    btn_ffmpeg_extract.click(fn=_btn_ffmpeg_extract, inputs=[video_file, extract_fmt, extract_fps], outputs=status_info_1, show_progress=False)

                gr.HTML(html.escape(r'=> expected to get ffprobe.json, task.json, frames\*.jpg, audio.wav'))

        with gr.Blocks():
            with gr.Tab('2: Make latents, masks, tags & conds'):
                status_info_2 = gr.HTML()

                with gr.Row().style(equal_height=True):
                    btn_latent = gr.Button('Make latent images!')
                    btn_latent.click(fn=_btn_latent, outputs=status_info_2, show_progress=False)

                    btn_midas = gr.Button('Make depth masks!')
                    btn_midas.click(fn=_btn_midas, outputs=status_info_2, show_progress=False)

                    btn_deepdanbooru = gr.Button('Make inverted tags!')
                    btn_deepdanbooru.click(fn=_btn_deepdanbooru, outputs=status_info_2, show_progress=False)

                with gr.Group():
                    gr.HTML(value=INVCOND_HELP_HTML)

                    with gr.Row().style(equal_height=True):
                        ic_prompt   = gr.Text  (label=LABEL_IC_PROMPT,   value=lambda: DEFAULT_IC_PROMPT, lines=2)
                    with gr.Row().style(equal_height=True):
                        ic_step     = gr.Slider(label=LABEL_IC_STEP,     value=lambda: DEFAULT_IC_STEP,     minimum=1, maximum=1000, step=1)
                        ic_grad_acc = gr.Slider(label=LABEL_IC_GRAD_ACC, value=lambda: DEFAULT_IC_GRAD_ACC, minimum=1, maximum=16, step=1)
                        ic_lr       = gr.Text  (label=LABEL_IC_LR,       value=lambda: DEFAULT_IC_LR,       max_lines=1)
                    with gr.Row().style(equal_height=True):
                        btn_invert_cond = gr.Button('Make inverted conditions!')
                        btn_invert_cond.click(fn=_btn_invert_cond, inputs=[ic_prompt, ic_step, ic_grad_acc, ic_lr], outputs=status_info_2, show_progress=False)

                gr.HTML(html.escape(r'=> expected to get latent\*.t, depthmask\*.png, tags.json, tags-topk.txt, icond\*.t'))

        with gr.Blocks():
            with gr.Tab('3: Prompt travel'):
                gr.HTML(value=PTRAVEL_HELP_HTML)

                with gr.Row(variant='compact'):
                    travel_ratio = gr.Slider(label=LABEL_TRAVEL_RATIO, value=lambda: DEFAULT_TRAVEL_RATIO, minimum=1,   maximum=10,  step=0.1)
                    travel_damp  = gr.Slider(label=LABEL_TRAVEL_DAMP,  value=lambda: DEFAULT_TRAVEL_DAMP,  minimum=1.0, maximum=2.0, step=0.1)
                    init_noise_w = gr.Slider(label=LABEL_INIT_NOISE_W, value=lambda: DEFAULT_INIT_NOISE_W, minimum=0.0, maximum=1.5, step=0.01)
                with gr.Row(variant='compact'):
                    use_mask    = gr.Checkbox(label=LABEL_USE_MASK, value=lambda: DEFAULT_USE_MASK)
                    mask_lowcut = gr.Slider(label=LABEL_MASK_LOWCUT, value=lambda: DEFAULT_MASK_LOWCUT, minimum=0, maximum=255, step=1)

                gr.HTML(html.escape(r'=> expected to get ptravel\*.png'))

        with gr.Blocks():
            with gr.Tab('4: Upscale & interpolate'):
                status_info_4 = gr.HTML()

                with gr.Row(variant='compact').style(equal_height=True):
                    resr_model = gr.Dropdown(label=LABEL_RESR_MODEL, value=lambda: DEFAULT_RESR_MODEL, choices=CHOICES_RESR_MODEL)
                    resr_ratio = gr.Dropdown(label=LABEL_RESR_RATIO, value=lambda: DEFAULT_RESR_RATIO, choices=CHOICES_RESR_RATIO)
                    btn_resr = gr.Button('Launch super-resolution!')
                    btn_resr.click(fn=_btn_resr, inputs=[resr_model, resr_ratio, extract_fmt], outputs=status_info_4, show_progress=False)

                with gr.Row(variant='compact').style(equal_height=True):
                    rife_model = gr.Dropdown(label=LABEL_RIFE_MODEL, value=lambda: DEFAULT_RIFE_MODEL, choices=CHOICES_RIFE_MODEL)
                    rife_fps = gr.Slider(label=LABEL_RIFE_FPS, value=lambda: DEFAULT_RIFE_FPS, minimum=12, maximum=60, step=1.0)
                    btn_rife = gr.Button('Launch frame-interpolation!')
                    btn_rife.click(fn=_btn_rife, inputs=[rife_model, rife_fps, extract_fmt, extract_fps, travel_ratio], outputs=status_info_4, show_progress=False)
                
                gr.HTML(html.escape(r'=> expected to get resr\*.jpg, rife\*.jpg'))

        with gr.Blocks():
            with gr.Tab('5: Render'):
                status_info_5 = gr.HTML()

                with gr.Row(variant='compact').style(equal_height=True):
                    export_fmt = gr.Dropdown(label=LABEL_EXPORT_FMT, value=lambda: DEFAULT_EXPORT_FMT, choices=CHOICES_VIDEO_FMT)
                    compose_src = gr.Dropdown(label=LABEL_COMPOSE_SRC, value=lambda: DEFAULT_COMPOSE_SRC, choices=CHOICES_COMPOSE_SRC)
                    btn_ffmpeg_compose = gr.Button('Render!')
                    btn_ffmpeg_compose.click(fn=_btn_ffmpeg_compose, inputs=[export_fmt, rife_fps, extract_fmt, compose_src], outputs=status_info_5, show_progress=False)

                gr.HTML(html.escape(r'=> expected to get synth.mp4'))

        with gr.Row(variant='compact'):
            allow_overwrite = gr.Checkbox(label=LABEL_ALLOW_OVERWRITE, value=lambda: DEFAULT_ALLOW_OVERWRITE)
            allow_overwrite.change(fn=_chk_overwrite, inputs=allow_overwrite)

            btn_interrut = gr.Button('Interrupt!', variant='primary')
            btn_interrut.click(fn=state.interrupt, show_progress=False)

        return [
            init_noise_w, travel_ratio, travel_damp, 
            use_mask, mask_lowcut
        ]

    def run(self, p:StableDiffusionProcessingImg2Img, 
            init_noise_w:float, travel_ratio:int, travel_damp:float, 
            use_mask:bool, mask_lowcut:int):

        if 'check cache exists':
            out_dp = workspace / WS_TRAVEL
            if out_dp.exists():
                if not cur_allow_overwrite:
                    return Processed(p, [], p.seed, task_ignore_str('ptrvale'))
                shutil.rmtree(str(out_dp))
            out_dp.mkdir()

        if 'check required materials exist':
            frames_dp = workspace / WS_FRAMES
            if not frames_dp.exists():
                return Processed(p, [], p.seed, f'frames folder not found: {frames_dp}')

            cond_dp = workspace / WS_ICOND
            if not cond_dp.exists():
                return Processed(p, [], p.seed, f'icond folder not found: {cond_dp}')

            mask_dp = workspace / WS_MASK
            if use_mask and not mask_dp.exists():
                return Processed(p, [], p.seed, f'mask folder not found: {mask_dp}')

        if 'check material integrity':
            n_inits = get_folder_file_count(frames_dp)
            if n_inits < 2:
                return Processed(p, [], p.seed, f'too few frames to go: n_frames ({n_inits})')
            
            n_conds = get_folder_file_count(cond_dp)
            if n_inits != n_conds:
                return Processed(p, [], p.seed, f'number mismatch for n_frames ({n_inits}) != n_conds ({n_conds})')

            if use_mask:
                n_masks = get_folder_file_count(mask_dp)
                if n_inits != n_masks:
                    return Processed(p, [], p.seed, f'number mismatch for n_frames ({n_inits}) != n_masks ({n_masks})')

        if 'override & fix p settings':
            p.n_iter              = 1
            p.batch_size          = 1
            p.seed                = get_fixed_seed(p.seed)
            p.subseed             = get_fixed_seed(p.subseed)
            p.do_not_save_grid    = True
            p.do_not_save_samples = False
            p.outpath_samples     = str(out_dp)
            p.initial_noise_multiplier = init_noise_w

        if 'update task config':
            cur_config.travel_ratio = travel_ratio
            cur_config.travel_damp  = travel_damp
            cur_config.use_mask     = use_mask
            cur_config.mask_lowcut  = mask_lowcut

            cur_config.i2i_negative_prompt    = p.negative_prompt
            cur_config.i2i_resize_mode        = p.resize_mode
            cur_config.i2i_sampler_name       = p.sampler_name
            cur_config.i2i_steps              = p.steps
            cur_config.i2i_width              = p.width
            cur_config.i2i_height             = p.height
            cur_config.i2i_cfg_scale          = p.cfg_scale
            cur_config.i2i_denoising_strength = p.denoising_strength
            cur_config.i2i_seed               = p.seed
            cur_config.i2i_subseed            = p.subseed
            cur_config.i2i_subseed_strength   = p.subseed_strength

            cur_config.save()

        if 'decide interp steps between stages':
            init_fns = os.listdir(frames_dp)

            # interp constantly `travel_ratio-1` frames between stages
            steps = np.ones(shape=[len(init_fns) - 1]) * travel_ratio - 1

            # interp with damping:
            #  |travel_damp| => 区间帧率放缩峰值倍数
            #  跳跃的帧间插帧更多，平滑的帧间插帧更少；应该会加重慢动作效果 🤔
            if travel_damp > 1.0:
                if 'calc dists between frames':
                    dist_fp = workspace / WS_FRAMES_DIST

                    if not dist_fp.exists():
                        print('>> calc dist and make cache file..')

                        def get_chan_wise_norm_img(fp: Path) -> np.ndarray:
                            img = Image.open(fp).convert('RGB')
                            im = np.asarray(img, dtype=np.float16) / 255.0  # [H, W, C]
                            std = im.std (axis=(0, 1), keepdims=True)
                            if std.mean() < np.finfo(np.float16).eps:
                                return np.zeros_like(im)
                            avg = im.mean(axis=(0, 1), keepdims=True)
                            return (im - avg) / std

                        L1 = []
                        im0, im1 = None, get_chan_wise_norm_img(frames_dp / init_fns[0])
                        for fn in tqdm(init_fns[1:]):
                            im0, im1 = im1, get_chan_wise_norm_img(frames_dp / fn)
                            L1.append(np.abs(im0 - im1).mean())     # 归一化像素距离：对线条运动敏感，对色变光照不敏感
                        L1 = np.asarray(L1)
                        del im0, im1, get_chan_wise_norm_img

                        N_SIGMA = 2
                        avg, std = L1.mean(), L1.std()
                        L1_clip = L1.clip(avg - N_SIGMA * std, avg + N_SIGMA * std)
                        del N_SIGMA, avg, std

                        with open(dist_fp, 'wb') as fh:
                            pkl.dump((L1, L1_clip), fh)
                    
                    with open(dist_fp, 'rb') as fh:
                        L1, L1_clip = pkl.load(fh)

                    del dist_fp

                if any(np.isnan(L1_clip)):
                    dump_fp = workspace / WS_BUG_DUMP
                    with open(dump_fp, 'wb') as fh:
                        data = (L1, L1_clip)
                        pkl.dump(data, fh)
                    print('>> found nan in dists, force default to case travel_damp=1.0 !! :(')
                    print(f'>> please report bug with file {dump_fp} to me :)')
                else:
                    def get_factor(d:np.array) -> np.array:
                        d = (d - d.min()) / (d.max() - d.min())  # => [0, 1]
                        d = 2 * d - 1                            # => [-1, 1]
                        return travel_damp ** d                  # => [F^-1, F^1]

                    tgt_n_frames = sum(steps) ; print('<< tgt_n_frames:', tgt_n_frames)
                    # rescale by factor
                    steps_frac = steps * get_factor(L1_clip)
                    print('<< cur_n_frames (scale):', sum(steps_frac))
                    steps_new = np.round(steps_frac)
                    cur_n_frames = sum(steps_new) ; print('<< cur_n_frames (round):', cur_n_frames)
                    if cur_n_frames != tgt_n_frames:
                        # slide average
                        steps_new = np.asarray([steps_new[0], *[(steps_new[i-1] + steps_new[i+1]) / 2 for i in range(1, len(steps_new)-1)], steps_new[-1]])
                        cur_n_frames = sum(steps_new) ; print('<< cur_n_frames (avgpool):', cur_n_frames)
                        # rescale to match tgt
                        steps_frac = steps_new / (cur_n_frames / tgt_n_frames)
                        print('<< cur_n_frames (rescale):', sum(steps_frac))
                        steps_new = np.round(steps_frac)
                        print('<< cur_n_frames (round):', sum(steps_new))

                steps = steps_new
                del steps_frac, steps_new, tgt_n_frames, cur_n_frames

                if 'savefig for debug':
                    # matplotlib color ref: https://matplotlib.org/stable/gallery/color/named_colors.html
                    import matplotlib.pyplot as plt

                    fig, (ax1, ax2) = plt.subplots(2, 1)
                    ax1.        plot(L1,      c='skyblue', label='L1')
                    ax1.twinx().plot(L1_clip, c='red',     label='L1 (clipped)')
                    ax1.set_title('distance')
                    ax2.plot(steps)
                    ax2.set_title('interpolation steps')
                    fig.legend()
                    fig.savefig(str(workspace / WS_TRAVEL_STEPS), dpi=400)
                    del fig, ax1, ax2, L1, L1_clip

            steps = steps.astype(int).tolist()
            gc.collect()

        self.init_dp     = frames_dp
        self.cond_dp     = cond_dp
        self.init_fns    = init_fns
        self.mask_dp     = mask_dp if use_mask else None
        self.mask_lowcut = mask_lowcut
        self.steps       = [0, *steps]     # interpolated frames count only

        def image_save_hijack(param:ImageSaveParams):
            # allow path length more than 260 chars...
            #param.filename = '\\\\?\\' + param.filename
            # make things short
            dp, fn = os.path.dirname(param.filename), os.path.basename(param.filename)
            name, ext = os.path.splitext(fn)
            param.filename = os.path.join(dp, name[:5] + ext)     # just 5-digit serial number

        def cfg_denoiser_hijack(param:CFGDenoiserParams):
            import matplotlib ; matplotlib.use('agg', force=True)
            import matplotlib.pyplot as plt

            #param.sigma = (param.sigma * 0 + 1) * 0.7
            print('>> step:', param.sampling_step)
            print('>> sigma:', param.sigma[-1].item())

            if 'show x':
                x = param.x[-1]
                tmp = x.squeeze().detach().cpu().numpy()
                plt.clf()
                plt.subplot(221) ; plt.imshow(tmp[0])
                plt.subplot(222) ; plt.imshow(tmp[1])
                plt.subplot(223) ; plt.imshow(tmp[2])
                plt.subplot(224) ; plt.imshow(tmp[3])
                plt.suptitle('x')
                plt.savefig(Path(r'D:\Desktop\test') / f'{param.sampling_step}.png')

        on_cfg_denoiser(cfg_denoiser_hijack)
        on_before_image_saved(image_save_hijack)
        process_images_before(p)
        images, info = self.run_linear(p)
        process_images_after(p)
        remove_callbacks_for_function(image_save_hijack)
        remove_callbacks_for_function(cfg_denoiser_hijack)

        # show only partial results
        return Processed(p, images[::10][:100], p.seed, info)

    def run_linear(self, p: StableDiffusionProcessingImg2Img) -> Tuple[List[PILImage], str]:
        interp_fn   = weighted_sum
        init_dp     = self.init_dp
        init_fns    = self.init_fns
        cond_dp     = self.cond_dp
        mask_dp     = self.mask_dp
        mask_lowcut = self.mask_lowcut
        steps       = self.steps
        n_stages    = len(init_fns)
        n_frames    = n_stages + sum(steps)  # = keyframes + interpolated

        initial_info: str = None
        images: List[PILImage] = []

        def get_icond(idx:int) -> Tensor:
            t: Tensor = torch.load(cond_dp / Path(init_fns[idx]).with_suffix('.t').name)
            t = t.squeeze(dim=0).to(p.sd_model.device)
            return t.detach()      # make deepcopy happy :)

        def get_init_img(idx:int) -> List[PILImage]:
            return [Image.open(init_dp / init_fns[idx]).convert('RGB')]

        def get_mask_img(idx:int) -> PILImage:
            if not mask_dp: return None

            def renorm_mask(im:np.ndarray, thresh:int) -> np.ndarray:
                # map under thresh to 0, renorm above thresh to [0.0, 1.0]
                mask_v = im >= thresh
                im_v = im * mask_v
                vmin, vmax = im_v.min(), im_v.max()
                im = (im.astype(np.float16) - vmin) / (vmax - vmin)
                im *= mask_v
                im = (im * np.iinfo(np.uint8).max).astype(np.uint8)
                return im

            img = Image.open(mask_dp / Path(init_fns[idx]).with_suffix('.png')).convert('L')
            im = np.asarray(img, dtype=np.uint8)
            im = renorm_mask(im, mask_lowcut)
            return Image.fromarray(im)

        def gen_image(pos_hidden, neg_hidden, prompts, seeds, subseeds):
            nonlocal images, initial_info, p
            proc = process_images_cond_to_image(p, pos_hidden, neg_hidden, prompts, seeds, subseeds)
            if initial_info is None: initial_info = proc.info
            images.extend(proc.images)

        # Step 1: draw the init image
        p.init_images = get_init_img(0)
        p.image_mask  = get_mask_img(0)
        print(f'[stage 1/{n_stages}] steps: 0')
        from_pos_hidden, neg_hidden, prompts, seeds, subseeds = process_images_prompt_to_cond(p)
        from_pos_hidden = cond_replace(from_pos_hidden, get_icond(0))
        gen_image(from_pos_hidden, neg_hidden, prompts, seeds, subseeds)

        # travel through stages
        i_frames = 1
        for i in range(1, n_stages):
            if state.interrupted: break

            state.job = f'{i_frames}/{n_frames}'
            state.job_no = i_frames + 1
            i_frames += 1

            # change reference materials
            p.init_images = get_init_img(i)
            p.image_mask  = get_mask_img(i)     # TODO: interp on mask?
            print(f'[stage {i+1}/{n_stages}] steps: {steps[i]}')
            to_pos_hidden, neg_hidden, prompts, seeds, subseeds = process_images_prompt_to_cond(p)
            to_pos_hidden = cond_replace(to_pos_hidden, get_icond(i))

            # Step 2: draw the interpolated images
            is_break_iter = False
            n_inter = steps[i] + 1
            for t in range(1, n_inter):
                if state.interrupted: is_break_iter = True ; break

                alpha = t / n_inter     # [1/T, 2/T, .. T-1/T]
                inter_pos_hidden = interp_fn(from_pos_hidden, to_pos_hidden, alpha)
                gen_image(inter_pos_hidden, neg_hidden, prompts, seeds, subseeds)

            if is_break_iter: break

            # Step 3: draw the fianl stage
            gen_image(to_pos_hidden, neg_hidden, prompts, seeds, subseeds)
            
            # move to next stage
            from_pos_hidden = to_pos_hidden

        return images, initial_info
