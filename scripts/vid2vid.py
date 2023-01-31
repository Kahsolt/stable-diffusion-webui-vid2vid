import os
import sys
import json
import html
import shutil
from re import compile as Regex
from pathlib import Path
from subprocess import Popen
from PIL import Image
from PIL.Image import Image as PILImage
from enum import Enum
from collections import Counter
from typing import List, Tuple, Dict, Callable, Union, Any
from traceback import print_exc, format_exc
from time import time
import gc

import gradio as gr
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from numpy.typing import NDArray
import cv2

from modules.scripts import Script
from modules.script_callbacks import remove_callbacks_for_function, on_before_image_saved, ImageSaveParams, on_cfg_denoiser, CFGDenoiserParams
from modules.ui import gr_show
from modules.devices import torch_gc, autocast, device, cpu
from modules.shared import opts, state
from modules.processing import Processed, StableDiffusionProcessingImg2Img, get_fixed_seed, process_images_inner
from modules.images import resize_image

try:
    # should be <sd-webui> root abspath
    SD_WEBUI_PATH = Path.cwd()
    # prompt-travel
    PTRAVEL_PATH = Path(SD_WEBUI_PATH) / 'extensions' / 'stable-diffusion-webui-prompt-travel'
    assert PTRAVEL_PATH.exists() ; sys.path.insert(0, str(PTRAVEL_PATH))
    from scripts.prompt_travel import process_images_before, process_images_after
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

class Img2ImgMode(Enum):
    BATCH  = 'batch img2img'
    SINGLE = 'img2img (for debug)'

class NoiseSched(Enum):
    DEFAULT  = '(default)'
    KARRAS   = 'karras'
    EXP      = 'exponential'
    POLY_EXP = 'poly-exponential'
    VP       = 'vp'
    LINEAR   = 'linear'

class FrameDeltaCorrection(Enum):
    NONE = 'none'
    AVG  = 'avg'
    STD  = 'std'
    NORM = 'avg & std'

def get_resr_model_names() -> List[str]:
    return sorted({fn.stem for fn in (RESR_PATH / 'models').iterdir()})

def get_rife_model_names() -> List[str]:
    #return [fn.name for fn in RIFE_PATH.iterdir() if fn.is_dir()]
    return ['rife-v4']      # TODO: `only rife-v4 model support custom numframe and timestep`

if 'global consts':
    # cache folder layout
    WS_FFPROBE              = 'ffprobe.json'
    WS_FRAMES               = 'frames'
    WS_AUDIO                = 'audio.wav'
    WS_DFRAME               = 'framedelta'
    WS_MASK                 = 'depthmask'
    WS_TAGS                 = 'tags.json'
    WS_IMG2IMG              = 'img2img'
    WS_IMG2IMG_DEBUG        = 'img2img_debug'
    WS_RESR                 = 'resr'
    WS_RIFE                 = 'rife'
    WS_SYNTH                = 'synth'    # stem

    WS_TAGS_TOPK            = 'tags-topk.txt'

    __ = lambda key, value=None: opts.data.get(f'customscript/vid2vid.py/img2img/{key}/value', value)

    LABEL_CACHE_FOLDER      = 'Cache Folder'
    LABEL_WORKSPACE_FOLDER  = 'Workspace Folder'
    LABEL_VIDEO_FILE        = 'Input video file'
    LABEL_VIDEO_INFO        = 'Video media info'
    LABEL_EXTRACT_FMT       = 'Extracted file format'
    LABEL_EXTRACT_FPS       = 'Extracted FPS'
    LABEL_IMG2IMG_MODE      = 'Img2Img mode'
    LABEL_INIT_NOISE_W      = 'Init noise weight'
    LABEL_SIGMA_OVERRIDE    = 'Override noise schedule'
    LABEL_SIGMA_METH        = 'Sigma method'
    LABEL_SIGMA_MAX         = 'Sigma max'
    LABEL_SIGMA_MIN         = 'Sigma min'
    LABEL_STEPS             = 'Sampling steps (override)'
    LABEL_DENOISE_W         = 'Denoising strength (override)'
    LABEL_FDC_METH          = 'Frame delta correction'
    LABEL_MASK_LOWCUT       = 'Depth mask low-cut'
    LABEL_RESR_MODEL        = 'Real-ESRGAN model'
    LABEL_RIFE_MODEL        = 'RIFE model'
    LABEL_RIFE_FPS          = 'Interpolated FPS for export'
    LABEL_EXPORT_FMT        = 'Export format'
    LABEL_COMPOSE_SRC       = 'Frame source'
    LABEL_ALLOW_OVERWRITE   = 'Allow overwrite cache'

    CHOICES_IMAGE_FMT       = [x.value for x in ImageFormat]
    CHOICES_VIDEO_FMT       = [x.value for x in VideoFormat]
    CHOICES_SIGMA_METH      = [x.value for x in NoiseSched]
    CHOICES_FDC_METH        = [x.value for x in FrameDeltaCorrection]
    CHOICES_IMG2IMG_MODE    = [x.value for x in Img2ImgMode]
    CHOICES_RESR_MODEL      = get_resr_model_names()
    CHOICES_RIFE_MODEL      = get_rife_model_names()
    CHOICES_COMPOSE_SRC     = [
        WS_FRAMES,
        WS_MASK,
        WS_DFRAME,
        WS_IMG2IMG,
        WS_RESR,
        WS_RIFE,
    ]

    INIT_CACHE_FOLDER = Path(os.environ['TMP']) / 'sd-webui-vid2vid'
    INIT_CACHE_FOLDER.mkdir(exist_ok=True)

    DEFAULT_CACHE_FOLDER    = __(LABEL_CACHE_FOLDER, str(INIT_CACHE_FOLDER))
    DEFAULT_EXTRACT_FMT     = __(LABEL_EXTRACT_FMT, ImageFormat.JPG.value)
    DEFAULT_EXTRACT_FPS     = __(LABEL_EXTRACT_FPS, 12)
    DEFAULT_IMG2IMG_MODE    = __(LABEL_IMG2IMG_MODE, Img2ImgMode.BATCH.value)
    DEFAULT_INIT_NOISE_W    = __(LABEL_INIT_NOISE_W, 0.95)
    DEFAULT_SIGMA_OVERRIDE  = __(LABEL_SIGMA_OVERRIDE, False)
    DEFAULT_SIGMA_METH      = __(LABEL_SIGMA_METH, NoiseSched.EXP.value)
    DEFAULT_SIGMA_MAX       = __(LABEL_SIGMA_MAX, 1.2)
    DEFAULT_SIGMA_MIN       = __(LABEL_SIGMA_MIN, 0.1)
    DEFAULT_STEPS           = __(LABEL_STEPS, 20)
    DEFAULT_DENOISE_W       = __(LABEL_DENOISE_W, 0.85)    
    DEFAULT_FDC_METH        = __(LABEL_FDC_METH, FrameDeltaCorrection.NORM.value)
    DEFAULT_MASK_LOWCUT     = __(LABEL_MASK_LOWCUT, -1)
    DEFAULT_RESR_MODEL      = __(LABEL_RESR_MODEL, 'realesr-animevideov3-x2')
    DEFAULT_RIFE_MODEL      = __(LABEL_RIFE_MODEL, 'rife-v4')
    DEFAULT_RIFE_FPS        = __(LABEL_RIFE_FPS, 24)
    DEFAULT_COMPOSE_SRC     = __(LABEL_COMPOSE_SRC, WS_RIFE)
    DEFAULT_EXPORT_FMT      = __(LABEL_EXPORT_FMT, VideoFormat.MP4.value)
    DEFAULT_ALLOW_OVERWRITE = __(LABEL_ALLOW_OVERWRITE, True)

    IMG2IMG_HELP_HTML = '''
<div>
  <h4> Instructions for this step: 😉 </h4>
  <p> 1. check settings below, and also <strong>all img2img settings</strong> top above ↑↑: prompts, sampler, size, seed, etc.. </p>
  <p> 2. use <strong>extra networks</strong> or <strong>embeddings</strong> as you want </p>
  <p> 3. <strong>select a dummy init image</strong> in the img2img ref-image box to avoid webui's "AttributeError" error :) </p>
  <p> 4. click the top-right master <strong>"Generate"</strong> button to go! </p>
</div>
'''


def get_folder_file_count(dp:Union[Path, str]) -> int:
    return len(os.listdir(dp))

def get_file_size(fp:Union[Path, str]) -> float:
    return os.path.getsize(fp) / 2**20


# global runtime vars
workspace: Path = None
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
        return Path(name).stem

    return Path(cache_folder) / safe_for_path(fn)

def _file_select(video_file:object) -> List[GradioRequest]:
    global workspace, cur_cache_folder

    # close workspace
    if video_file is None:
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
            print(f'>> run task {cur_task[5:]!r}')      # remove '_btn_'
            state.interrupted = False
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
    out_dp.mkdir()

    out_fp = workspace / WS_AUDIO
    if out_fp.exists(): out_fp.unlink()

    try:
        cmd = f'"{FFMPEG_BIN}" -i "{video_file.name}" -an -r {extract_fps} -f image2 -q:v 2 {out_dp}\\%05d.{extract_fmt}'
        print(f'>> exec: {cmd}')
        Popen(cmd, shell=True, text=True, encoding='utf-8').wait()
        try:
            cmd = f'"{FFMPEG_BIN}" -i "{video_file.name}" -vn "{out_fp}"'
            Popen(cmd, shell=True, text=True, encoding='utf-8').wait()
            has_audio = 'yes'
        except:
            has_audio = 'no'
            print_exc()

        return RetCode.INFO, f'frames: {get_folder_file_count(out_dp)}, audio: {has_audio}'
    except KeyboardInterrupt:
        return RetCode.WARN, 'interrupted by Ctrl+C'
    except:
        e = format_exc() ; print(e)
        return RetCode.ERROR, e

@task
def _btn_frame_delta() -> TaskResponse:
    in_dp = workspace / WS_FRAMES
    if not in_dp.exists():
        return RetCode.ERROR, f'frames folder not found: {in_dp}'

    out_dp = workspace / WS_DFRAME
    if out_dp.exists():
        if not cur_allow_overwrite:
            return RetCode.WARN, task_ignore_str('framedelta')
        shutil.rmtree(str(out_dp))
    out_dp.mkdir()

    try:
        def get_img(fp: Path) -> NDArray[np.float16]:
            img = Image.open(fp).convert('RGB')
            return np.asarray(img, dtype=np.float16) / 255.0  # [H, W, C]

        fps = list(in_dp.iterdir())
        im0, im1 = None, get_img(fps[0])
        for fp in tqdm(fps[1:]):
            if state.interrupted: break

            im0, im1 = im1, get_img(fp)
            d = im1 - im0           # [-1, 1]
            d_n = (d + 1) / 2       # [0, 1]
            d_i = (d_n * np.iinfo(np.uint8).max).astype(np.uint8)  # [0, 255]
            img = Image.fromarray(d_i)
            img.save(out_dp / f'{fp.stem}.png')

        return RetCode.INFO, f'framedelta: {get_folder_file_count(out_dp)}'
    except KeyboardInterrupt:
        return RetCode.WARN, 'interrupted by Ctrl+C'
    except:
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
                img_out.save(out_dp / f'{Path(fn).stem}.png')

        return RetCode.INFO, f'frames: {get_folder_file_count(out_dp)}'
    except KeyboardInterrupt:
        return RetCode.WARN, 'interrupted by Ctrl+C'
    except:
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
    except KeyboardInterrupt:
        return RetCode.WARN, 'interrupted by Ctrl+C'
    except:
        e = format_exc() ; print(e)
        return RetCode.ERROR, e
    finally:
        deepbooru_model.model.to(cpu)
        torch_gc()
        gc.collect()

@task
def _btn_resr(resr_model:str) -> TaskResponse:
    in_dp = workspace / WS_IMG2IMG
    if not in_dp.exists():
        return RetCode.ERROR, f'img2img folder not found: {in_dp}'

    out_dp = workspace / WS_RESR
    if out_dp.exists():
        if not cur_allow_overwrite:
            return RetCode.WARN, task_ignore_str('resr')
        shutil.rmtree(str(out_dp))
    out_dp.mkdir()

    try:
        try:
            m = Regex('-x(\d)').search(resr_model).groups()
            resr_ratio = int(m[0])
        except:
            print('>> cannot parse `resr_ratio` form model name, defaults to 2')
            resr_ratio = 2

        cmd = f'"{RESR_BIN}" -v -s {resr_ratio} -n {resr_model} -i "{in_dp}" -o "{out_dp}"'
        print(f'>> exec: {cmd}')
        Popen(cmd, shell=True, text=True, encoding='utf-8').wait()

        return RetCode.INFO, f'frames: {get_folder_file_count(out_dp)}'
    except KeyboardInterrupt:
        return RetCode.WARN, 'interrupted by Ctrl+C'
    except:
        e = format_exc() ; print(e)
        return RetCode.ERROR, e

@task
def _btn_rife(rife_model:str, rife_fps:float, extract_fmt:str, extract_fps:float) -> TaskResponse:
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
        n_interp = get_folder_file_count(workspace / WS_FRAMES) * rife_fps // extract_fps
        cmd = f'"{RIFE_BIN}" -v -n {n_interp} -m {rife_model} -f %05d.{extract_fmt} -i "{in_dp}" -o "{out_dp}"'
        print(f'>> exec: {cmd}')
        Popen(cmd, shell=True, text=True, encoding='utf-8').wait()

        return RetCode.INFO, f'frames: {get_folder_file_count(out_dp)}'
    except KeyboardInterrupt:
        return RetCode.WARN, 'interrupted by Ctrl+C'
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

    fn_ext = 'png' if src in [WS_DFRAME, WS_MASK, WS_IMG2IMG, WS_RESR] else extract_fmt
    try:
        cmd = f'"{FFMPEG_BIN}"{opts} -framerate {rife_fps} -i "{in_img}\\%05d.{fn_ext}" -crf 20 -c:v libx264 -pix_fmt yuv420p "{out_vid}"'
        print(f'>> exec: {cmd}')
        Popen(cmd, shell=True, text=True, encoding='utf-8').wait()

        return RetCode.INFO, f'filesize: {get_file_size(out_vid):.3f}'
    except KeyboardInterrupt:
        return RetCode.WARN, 'interrupted by Ctrl+C'
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
                    extract_fmt = gr.Dropdown(label=LABEL_EXTRACT_FMT, value=lambda: DEFAULT_EXTRACT_FMT, choices=CHOICES_IMAGE_FMT)
                    extract_fps = gr.Slider(label=LABEL_EXTRACT_FPS, value=lambda: DEFAULT_EXTRACT_FPS, minimum=1, maximum=24, step=0.1)
                    
                    btn_ffmpeg_extract = gr.Button('Extract frames!')
                    btn_ffmpeg_extract.click(fn=_btn_ffmpeg_extract, inputs=[video_file, extract_fmt, extract_fps], outputs=status_info_1, show_progress=False)

                gr.HTML(html.escape(r'=> expected to get ffprobe.json, frames\*.jpg, audio.wav'))

        with gr.Blocks():
            with gr.Tab('2: Make masks & tags'):
                status_info_2 = gr.HTML()

                with gr.Row().style(equal_height=True):
                    btn_frame_delta = gr.Button('Make frame delta!')
                    btn_frame_delta.click(fn=_btn_frame_delta, outputs=status_info_2, show_progress=False)

                    btn_midas = gr.Button('Make depth masks!')
                    btn_midas.click(fn=_btn_midas, outputs=status_info_2, show_progress=False)

                    btn_deepdanbooru = gr.Button('Make inverted tags!')
                    btn_deepdanbooru.click(fn=_btn_deepdanbooru, outputs=status_info_2, show_progress=False)

                gr.HTML(html.escape(r'=> expected to get framedelta\*.png, depthmask\*.png, tags.json, tags-topk.txt'))

        with gr.Blocks():
            with gr.Tab('3: Successive Img2Img'):
                gr.HTML(value=IMG2IMG_HELP_HTML)

                with gr.Row(variant='compact'):
                    img2img_mode = gr.Radio(label=LABEL_IMG2IMG_MODE, value=lambda: DEFAULT_IMG2IMG_MODE, choices=CHOICES_IMG2IMG_MODE)

                with gr.Row(variant='compact'):
                    steps        = gr.Slider(label=LABEL_STEPS,        value=lambda: DEFAULT_STEPS,        minimum=1,   maximum=150, step=1)
                    denoise_w    = gr.Slider(label=LABEL_DENOISE_W,    value=lambda: DEFAULT_DENOISE_W,    minimum=0.0, maximum=1.0, step=0.01)
                    init_noise_w = gr.Slider(label=LABEL_INIT_NOISE_W, value=lambda: DEFAULT_INIT_NOISE_W, minimum=0.0, maximum=1.5, step=0.01)

                with gr.Row(variant='compact'):
                    sigma_meth = gr.Dropdown(label=LABEL_SIGMA_METH, value=lambda: DEFAULT_SIGMA_METH, choices=CHOICES_SIGMA_METH)
                    sigma_max  = gr.Slider  (label=LABEL_SIGMA_MAX,  value=lambda: DEFAULT_SIGMA_MAX,  minimum=0.0, maximum=5.0, step=0.01)
                    sigma_min  = gr.Slider  (label=LABEL_SIGMA_MIN,  value=lambda: DEFAULT_SIGMA_MIN,  minimum=0.0, maximum=5.0, step=0.01)

                def sigma_meth_change(sigma_meth):
                    show_param = NoiseSched(sigma_meth) != NoiseSched.DEFAULT
                    return [ gr_show(show_param), gr_show(show_param) ]
                sigma_meth.change(fn=sigma_meth_change, inputs=sigma_meth, outputs=[sigma_max, sigma_min], show_progress=False)

                with gr.Row(variant='compact') as tab_bacth_i2i:
                    fdc_methd   = gr.Dropdown(label=LABEL_FDC_METH,    value=lambda: DEFAULT_FDC_METH,    choices=CHOICES_FDC_METH)
                    mask_lowcut = gr.Slider  (label=LABEL_MASK_LOWCUT, value=lambda: DEFAULT_MASK_LOWCUT, minimum=-1,  maximum=255, step=1)

                def img2img_mode_change(img2img_mode:str):
                    is_show = Img2ImgMode(img2img_mode) == Img2ImgMode.BATCH
                    return gr_show(is_show)
                img2img_mode.change(fn=img2img_mode_change, inputs=img2img_mode, outputs=tab_bacth_i2i, show_progress=False)

                gr.HTML(html.escape(r'=> expected to get img2img\*.png'))

        with gr.Blocks():
            with gr.Tab('4: Upscale & interpolate'):
                status_info_4 = gr.HTML()

                with gr.Row(variant='compact').style(equal_height=True):
                    resr_model = gr.Dropdown(label=LABEL_RESR_MODEL, value=lambda: DEFAULT_RESR_MODEL, choices=CHOICES_RESR_MODEL)
                    btn_resr = gr.Button('Launch super-resolution!')
                    btn_resr.click(fn=_btn_resr, inputs=resr_model, outputs=status_info_4, show_progress=False)

                with gr.Row(variant='compact').style(equal_height=True):
                    rife_model = gr.Dropdown(label=LABEL_RIFE_MODEL, value=lambda: DEFAULT_RIFE_MODEL, choices=CHOICES_RIFE_MODEL)
                    rife_fps = gr.Slider(label=LABEL_RIFE_FPS, value=lambda: DEFAULT_RIFE_FPS, minimum=1, maximum=60, step=1.0)
                    btn_rife = gr.Button('Launch frame-interpolation!')
                    btn_rife.click(fn=_btn_rife, inputs=[rife_model, rife_fps, extract_fmt, extract_fps], outputs=status_info_4, show_progress=False)
                
                gr.HTML(html.escape(r'=> expected to get resr\*.jpg, rife\*.jpg'))

        with gr.Blocks():
            with gr.Tab('5: Render'):
                status_info_5 = gr.HTML()

                with gr.Row(variant='compact').style(equal_height=True):
                    export_fmt = gr.Dropdown(label=LABEL_EXPORT_FMT, value=lambda: DEFAULT_EXPORT_FMT, choices=CHOICES_VIDEO_FMT)
                    compose_src = gr.Dropdown(label=LABEL_COMPOSE_SRC, value=lambda: DEFAULT_COMPOSE_SRC, choices=CHOICES_COMPOSE_SRC)
                    btn_ffmpeg_compose = gr.Button('Render!')
                    btn_ffmpeg_compose.click(fn=_btn_ffmpeg_compose, inputs=[export_fmt, rife_fps, extract_fmt, compose_src], outputs=status_info_5, show_progress=False)

                gr.HTML(html.escape(r'=> expected to get synth-*.mp4'))

        with gr.Row(variant='compact'):
            allow_overwrite = gr.Checkbox(label=LABEL_ALLOW_OVERWRITE, value=lambda: DEFAULT_ALLOW_OVERWRITE)
            allow_overwrite.change(fn=_chk_overwrite, inputs=allow_overwrite)

            btn_interrut = gr.Button('Interrupt!', variant='primary')
            btn_interrut.click(fn=state.interrupt, show_progress=False)

        return [
            img2img_mode, fdc_methd, mask_lowcut,
            steps, denoise_w, init_noise_w,
            sigma_meth, sigma_max, sigma_min,
        ]

    def run(self, p:StableDiffusionProcessingImg2Img, 
            img2img_mode:str, fdc_methd:str, mask_lowcut:int,
            steps:int, denoise_w:float, init_noise_w:float, 
            sigma_meth:str, sigma_max:float, sigma_min:float, 
        ):

        if workspace is None:
            return Processed(p, [], p.seed, 'no current workspace opened!')

        img2img_mode = Img2ImgMode(img2img_mode)
        sigma_enable = NoiseSched(sigma_meth) != NoiseSched.DEFAULT

        if img2img_mode == Img2ImgMode.BATCH:
            use_mask = mask_lowcut >= 0

            if 'check cache exists':
                out_dp = workspace / WS_IMG2IMG
                if out_dp.exists():
                    if not cur_allow_overwrite:
                        return Processed(p, [], p.seed, task_ignore_str('img2img'))
                    shutil.rmtree(str(out_dp))
                out_dp.mkdir()

            if 'check required materials exist':
                frames_dp = workspace / WS_FRAMES
                if not frames_dp.exists():
                    return Processed(p, [], p.seed, f'frames folder not found: {frames_dp}')

                delta_dp = workspace / WS_DFRAME
                if not delta_dp.exists():
                    return Processed(p, [], p.seed, f'framedelta folder not found: {delta_dp}')

                mask_dp = workspace / WS_MASK
                if use_mask and not mask_dp.exists():
                    return Processed(p, [], p.seed, f'mask folder not found: {mask_dp}')

            if 'check material integrity':
                if use_mask:
                    n_inits = get_folder_file_count(frames_dp)
                    n_masks = get_folder_file_count(mask_dp)
                    if n_inits != n_masks:
                        return Processed(p, [], p.seed, f'number mismatch for n_frames ({n_inits}) != n_masks ({n_masks})')

            self.init_dp     = frames_dp
            self.init_fns    = os.listdir(frames_dp)
            self.delta_dp    = delta_dp
            self.mask_dp     = mask_dp if use_mask else None
            self.mask_lowcut = mask_lowcut
            self.fdc_methd   = FrameDeltaCorrection(fdc_methd)
        else:
            out_dp = workspace / WS_IMG2IMG_DEBUG
            out_dp.mkdir(exist_ok=True)

        if sigma_enable:
            from k_diffusion.sampling import get_sigmas_karras, get_sigmas_exponential, get_sigmas_polyexponential, get_sigmas_vp

            sigma_meth = NoiseSched(sigma_meth)
            sigma_min = max(sigma_min, 1e-3)

            if   sigma_meth == NoiseSched.KARRAS:
                sigma_fn = lambda n: get_sigmas_karras(n, sigma_min, sigma_max)
            elif sigma_meth == NoiseSched.EXP:
                sigma_fn = lambda n: get_sigmas_exponential(n, sigma_min, sigma_max)
            elif sigma_meth == NoiseSched.POLY_EXP:
                sigma_fn = lambda n: get_sigmas_polyexponential(n, sigma_min, sigma_max)
            elif sigma_meth == NoiseSched.VP:
                sigma_fn = lambda n: get_sigmas_vp(n, sigma_max, sigma_min)
            elif sigma_meth == NoiseSched.LINEAR:
                sigma_fn = lambda n: torch.linspace(sigma_max, sigma_min, n)

            p.steps = steps
            p.denoising_strength = denoise_w
            p.sampler_noise_scheduler_override = lambda step: sigma_fn(step).to(p.sd_model.device)

        if 'override & fix p settings':
            p.n_iter              = 1
            p.batch_size          = 1
            p.seed                = get_fixed_seed(p.seed)
            p.subseed             = get_fixed_seed(p.subseed)
            p.do_not_save_grid    = True
            p.do_not_save_samples = False
            p.outpath_samples     = str(out_dp)
            p.initial_noise_multiplier = init_noise_w

        def cfg_denoiser_hijack(param:CFGDenoiserParams):
            print(f'>> [{param.sampling_step}/{param.total_sampling_steps}] sigma: {param.sigma[-1].item()}')

        runner = self.run_batch_img2img if img2img_mode == Img2ImgMode.BATCH else self.run_img2img

        on_cfg_denoiser(cfg_denoiser_hijack)
        process_images_before(p)
        images, info = runner(p)
        process_images_after(p)
        remove_callbacks_for_function(cfg_denoiser_hijack)

        # show only partial results
        return Processed(p, images[::DEFAULT_EXTRACT_FPS][:100], p.seed, info)

    def run_img2img(self, p:StableDiffusionProcessingImg2Img) -> Tuple[List[PILImage], str]:
        proc = process_images_inner(p)
        return proc.images, proc.info

    def run_batch_img2img(self, p: StableDiffusionProcessingImg2Img) -> Tuple[List[PILImage], str]:
        init_dp     = self.init_dp
        init_fns    = self.init_fns
        delta_dp    = self.delta_dp
        mask_dp     = self.mask_dp
        mask_lowcut = self.mask_lowcut
        fdc_methd   = self.fdc_methd
        
        initial_info: str = None
        images: List[PILImage] = []

        def get_init(idx:int) -> List[PILImage]:
            return [Image.open(init_dp / init_fns[idx]).convert('RGB')]
        
        def get_dframe(idx:int, w:int, h:int) -> NDArray[np.float16]:
            img = Image.open(delta_dp / Path(init_fns[idx]).with_suffix('.png'))
            img = resize_image(p.resize_mode, img, w, h)
            im = np.asarray(img, dtype=np.float16) / 255.0  # [0, 1]
            im = im * 2 - 1                                 # [-1, 1]
            return im

        def get_mask(idx:int) -> PILImage:
            if not mask_dp: return None

            def renorm_mask(im:NDArray[np.uint8], thresh:int) -> NDArray[np.float16]:
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

        last_frame:NDArray[np.float16] = None
        iframe = 0
        def image_save_hijack(param:ImageSaveParams):
            # allow path length more than 260 chars...
            #param.filename = '\\\\?\\' + param.filename
            # just make things short
            dp, fn = os.path.dirname(param.filename), os.path.basename(param.filename)
            name, ext = os.path.splitext(fn)
            param.filename = os.path.join(dp, name[:5] + ext)     # just 5-digit serial number

            # frame delta consistency correction
            if fdc_methd != FrameDeltaCorrection.NONE:
                nonlocal last_frame, iframe

                def img_to_im(img: PILImage) -> NDArray[np.float16]:
                    return np.asarray(img.convert('RGB'), dtype=np.float16) / 255.0

                if last_frame is not None:
                    this_frame = img_to_im(param.image)
                    H, W, C = this_frame.shape
                    cur_d = this_frame - last_frame     # [-1.0, 1.0]
                    cur_d_t = torch.from_numpy(cur_d)
                    cur_avg = cur_d_t.mean(axis=[0, 1], keepdims=True).numpy()
                    cur_std = cur_d_t.std (axis=[0, 1], keepdims=True).numpy()
                    cur_d_n = (cur_d - cur_avg) / cur_std

                    tgt_d = get_dframe(iframe, W, H)    # [-1.0, 1.0]
                    tgt_d_t = torch.from_numpy(tgt_d)
                    tgt_avg = tgt_d_t.mean(axis=[0, 1], keepdims=True).numpy()
                    tgt_std = tgt_d_t.std (axis=[0, 1], keepdims=True).numpy()

                    if   fdc_methd == FrameDeltaCorrection.AVG:  new_d = cur_d_n * cur_std + tgt_avg
                    elif fdc_methd == FrameDeltaCorrection.STD:  new_d = cur_d_n * tgt_std + cur_avg
                    elif fdc_methd == FrameDeltaCorrection.NORM: new_d = cur_d_n * tgt_std + tgt_avg

                    im = (last_frame + new_d).clip(0.0, 1.0)
                    param.image = Image.fromarray((im * np.iinfo(np.uint8).max).astype(np.uint8)).convert('RGB')

                    last_frame = im
                else:
                    last_frame = img_to_im(param.image)

        on_before_image_saved(image_save_hijack)

        n_frames = len(init_fns)
        state.job_count = n_frames
        for i in tqdm(range(n_frames)):
            if state.interrupted: break

            state.job = f'{i}/{n_frames}'
            state.job_no = i + 1
            iframe = i

            p.init_images = get_init(i)
            p.image_mask  = get_mask(i)

            proc = process_images_inner(p)
            if initial_info is None: initial_info = proc.info
            images.extend(proc.images)

        remove_callbacks_for_function(image_save_hijack)

        return images, initial_info
