# stable-diffusion-webui-vid2vid

    Translate a video to some AI generated stuff, extension script for AUTOMATIC1111/stable-diffusion-webui.

----

<p align="left">
  <a href="https://github.com/Kahsolt/stable-diffusion-webui-vid2vid/commits"><img alt="Last Commit" src="https://img.shields.io/github/last-commit/Kahsolt/stable-diffusion-webui-vid2vid"></a>
  <a href="https://github.com/Kahsolt/stable-diffusion-webui-vid2vid/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/Kahsolt/stable-diffusion-webui-vid2vid"></a>
  <a href="https://github.com/Kahsolt/stable-diffusion-webui-vid2vid/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/Kahsolt/stable-diffusion-webui-vid2vid"></a>
  <a href="https://github.com/Kahsolt/stable-diffusion-webui-vid2vid/network"><img alt="GitHub forks" src="https://img.shields.io/github/forks/Kahsolt/stable-diffusion-webui-vid2vid"></a>
  <img alt="Language" src="https://img.shields.io/github/languages/top/Kahsolt/stable-diffusion-webui-vid2vid">
  <img alt="License" src="https://img.shields.io/github/license/Kahsolt/stable-diffusion-webui-vid2vid">
  <br/>
</p>

![:stable-diffusion-webui-vid2vid](https://count.getloli.com/get/@:stable-diffusion-webui-vid2vid)

Convert a video to an AI generated video through a pipeline of model neural models: Stable-Diffusion, DeepDanbooru, Midas, Real-ESRGAN, RIFE, etc. 

Although it sounds like the old joke that an English wizard turns a walnut into another walnut by reciting a tongue-twisting spell. 🤣  
Whatsoever, this synthesized videos should be essentially smoother than simple frame-to-frame img2img with post interpolation, due to the ability of semantical latent space interploation enabled by [prompt-travel](https://github.com/Kahsolt/stable-diffusion-webui-prompt-travel). 😉

Example: 

| original | rendered |
| :-: | :-: |
| ![original](img/original.webm) | ![rendered](img/rendered.webm) |


```
Sampler: Euler
Width: 832
Heigh: 448
Resize mode: Just Resize
Steps: 50                   (important! take a long process to anneal steadily, real_step ~= step * denoising_strength)
Denoising strength: 0.33    (important! take only tailing steps, usually < 0.5)
CFG Scale: 7
Seed: 114514

Extracted FPS: 8
Extracted fmt: jpg
InvCond init prompt: (masterpiece:1.3), highres, male_focus, 1boy, solo, indoors, looking_at_viewer, shirt, depth_of_field, blonde_hair, kagamine_len, black_collar, necktie, short_ponytail, spiked_hair, yellow_necktie, bass_clef, blue_eyes, headphones, white_shirt, sitting, collar, sailor_collar, short_sleeves, upper_body, brown_hair, short_hair, yellow_nails, headset, room
InvCond steps: 32
InvCond grad acc ratio: 1
InvCond lr: 0.001
Travel interp ratio: 1
Travel damping: 1.0
Init noise weight: 0.95
Use mask: false
RESR model: animevideov3
RESR ratio: 2
RIFE model: rife-v4
Interp/export FPS: 30
Export fmt: mp4
```

title:【LEN/MMD】和风模组面前耍大刀【青月/蓝铁/妖狐】
url: [https://www.bilibili.com/video/BV1Vd4y1L7Q9](https://www.bilibili.com/video/BV1Vd4y1L7Q9)
bid: BV1Vd4y1L7Q9
uploader: パンキッシュ


### How it works?

![How it works](img/How%20it%20works.png)

⚪ about override sigma schedule

**sigma schedule** controls how much noise is added to the latent image at each sampling step, so that the denoiser can do work against it -- it is a fight. 😃  
more noise will allow the denoiser to repaint the canvas, while less noise will result fuzzy-glasss-like image

- init noise weight: multipilier to the noise of init ref-image
- sigma method: various schdulers to create a serial of numbers decreasing in value
  - we recommend to try `linear` and `exponential` first to


### Installation

⚪ Auto install

- run `install.cmd`
- if you got any errors like `Access denied.`, try run it again until you see `Done!` without errors 😂

⚪ Maunal install

- install prompt-travel extension
  - `cd <SD>\extensions`
  - `git clone https://github.com/Kahsolt/stable-diffusion-webui-prompt-travel.git`
  - `cd stable-diffusion-webui-prompt-travel\tools`
  - run `install.cmd` to install the post-processing tools
- install MiDaS
  - `cd <SD>\repositories`
  - `git clone https://github.com/isl-org/MiDaS.git midas`
  - download `https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21_small-70d6b9c8.pt` put under `<SD>\models\midas\`


### Options

ℹ This script is only applicable in `Img2Img` tab :)  
⚠ some tasks will take a real long time, DO NOT click the button twice, juts see output on console!!

- cache_folder: (string), path to the folder for caching all intermediate data
- video_file: (file), path to the video file to convert



#### Acknowledgement

- FFmpeg: [https://ffmpeg.org](https://ffmpeg.org)
- MiDaS: [https://github.com/isl-org/MiDaS](https://github.com/isl-org/MiDaS)
- stable-diffuison-webui: [https://github.com/AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
  - TorchDeepDanbooru: [https://github.com/AUTOMATIC1111/TorchDeepDanbooru](https://github.com/AUTOMATIC1111/TorchDeepDanbooru)
  - depthmap2mask: [https://github.com/Extraltodeus/depthmap2mask](https://github.com/Extraltodeus/depthmap2mask)
  - prompt-travel: [https://github.com/Kahsolt/stable-diffusion-webui-prompt-travel](https://github.com/Kahsolt/stable-diffusion-webui-prompt-travel)
- Real-ESRGAN-ncnn-vulkan: [https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan)
- rife-ncnn-vulkan: [https://github.com/nihui/rife-ncnn-vulkan](https://github.com/nihui/rife-ncnn-vulkan)

----

by Armit
2023/01/20 
