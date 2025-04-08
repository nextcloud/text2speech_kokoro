import asyncio
import os
import io
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Event, Thread
from time import sleep, perf_counter
import logging
import soundfile as sf
import torch

from fastapi import FastAPI
from nc_py_api import NextcloudApp
from nc_py_api.ex_app import AppAPIAuthMiddleware, LogLvl, run_app, set_handlers
from nc_py_api.ex_app.providers.task_processing import TaskProcessingProvider, ShapeType, ShapeDescriptor, \
    ShapeEnumValue
from kokoro import KPipeline

VOICE_DESCRIPTIONS = {
    "af_heart": "Heart (American Female)",
    "af_alloy": "Alloy (American Female)",
    "af_aoede": "Aoede (American Female)",
    "af_bella": "Bella (American Female)",
    "af_jessica": "Jessica (American Female)",
    "af_kore": "Kore (American Female)",
    "af_nicole": "Nicole (American Female)",
    "af_nova": "Nova (American Female)",
    "af_river": "River (American Female)",
    "af_sarah": "Sarah (American Female)",
    "af_sky": "Sky (American Female)",
    "am_adam": "Adam (American Male)",
    "am_echo": "Echo (American Male)",
    "am_eric": "Eric (American Male)",
    "am_fenrir": "Fenrir (American Male)",
    "am_liam": "Liam (American Male)",
    "am_michael": "Michael (American Male)",
    "am_onyx": "Onyx (American Male)",
    "am_puck": "Puck (American Male)",
    "am_santa": "Santa (American Male)",
    "bf_alice": "Alice (British Female)",
    "bf_emma": "Emma (British Female)",
    "bf_isabella": "Isabella (British Female)",
    "bf_lily": "Lily (British Female)",
    "bm_daniel": "Daniel (British Male)",
    "bm_fable": "Fable (British Male)",
    "bm_george": "George (British Male)",
    "bm_lewis": "Lewis (British Male)",
    "jf_alpha": "Alpha (Japanese Female)",
    "jf_gongitsune": "Gongitsune (Japanese Female)",
    "jf_nezumi": "Nezumi (Japanese Female)",
    "jf_tebukuro": "Tebukuro (Japanese Female)",
    "jm_kumo": "Kumo (Japanese Male)",
    "zf_xiaobei": "Xiaobei (Mandarin Chinese Female)",
    "zf_xiaoni": "Xiaoni (Mandarin Chinese Female)",
    "zf_xiaoxiao": "Xiaoxiao (Mandarin Chinese Female)",
    "zf_xiaoyi": "Xiaoyi (Mandarin Chinese Female)",
    "zm_yunjian": "Yunjian (Mandarin Chinese Male)",
    "zm_yunxi": "Yunxi (Mandarin Chinese Male)",
    "zm_yunxia": "Yunxia (Mandarin Chinese Male)",
    "zm_yunyang": "Yunyang (Mandarin Chinese Male)",
    "ef_dora": "Dora (Spanish Female)",
    "em_alex": "Alex (Spanish Male)",
    "em_santa": "Santa (Spanish Male)",
    "ff_siwis": "Siwis (French Female)",
    "hf_alpha": "Alpha (Hindi Female)",
    "hf_beta": "Beta (Hindi Female)",
    "hm_omega": "Omega (Hindi Male)",
    "hm_psi": "Psi (Hindi Male)",
    "if_sara": "Sara (Italian Female)",
    "im_nicola": "Nicola (Italian Male)",
    "pf_dora": "Dora (Brazilian Portuguese Female)",
    "pm_alex": "Alex (Brazilian Portuguese Male)",
    "pm_santa": "Santa (Brazilian Portuguese Male)"
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def log(nc, level, content):
    logger.log((level + 1) * 10, content)
    if level < LogLvl.WARNING:
        return
    try:
        asyncio.run(nc.log(level, content))
    except:
        pass


TASKPROCESSING_PROVIDER_ID = 'text2speech_kokoro'


@asynccontextmanager
async def lifespan(_app: FastAPI):
    set_handlers(
        APP,
        enabled_handler
    )
    nc = NextcloudApp()
    if nc.enabled_state:
        app_enabled.set()
    start_bg_task()
    yield


APP = FastAPI(lifespan=lifespan)
APP.add_middleware(AppAPIAuthMiddleware)  # set global AppAPI authentication middleware

app_enabled = Event()


def background_thread_task():
    nc = NextcloudApp()
    while not app_enabled.is_set():
        sleep(5)
    print("Starting background task")
    pipes = {}  # Stores for each model a KPipeline instance

    while True:
        if not app_enabled.is_set():
            sleep(30)
            continue
        try:
            next_task = nc.providers.task_processing.next_task([TASKPROCESSING_PROVIDER_ID], ['core:text2speech'])
            if 'task' not in next_task or next_task is None:
                sleep(5)
                continue
            task = next_task.get('task')
        except Exception as e:
            print(str(e))
            log(nc, LogLvl.ERROR, str(e))
            sleep(30)
            continue
        try:
            log(nc, LogLvl.INFO, f"Next task: {task['id']}")
            prompt = task.get("input").get('input')
            voice = task.get("input").get('voice') or 'af_heart'  # Use 'af_heart' if voice is not specified
            lang_code = voice[0]
            speed = task.get("input").get('speed') or 1

            log(nc, LogLvl.INFO, "generating speech with voice: " + voice + " and speed: " + str(speed))
            time_start = perf_counter()
            pipe = pipes.get(lang_code)
            if pipe is None:
                pipe = KPipeline(lang_code=lang_code)
                pipes[lang_code] = pipe
            generator = pipe(prompt, voice=voice, speed=speed)
            speechs = []
            for _, _, speech in generator:
                speechs.append(speech)
            speech = torch.cat(speechs, dim=0)
            log(nc, LogLvl.INFO, f"speech generated: {perf_counter() - time_start}s")

            speech_stream = io.BytesIO()
            speech_stream.name = 'speech.wav'
            sf.write(speech_stream, speech, 24000)
            speech_id = nc.providers.task_processing.upload_result_file(task.get('id'), speech_stream)

            NextcloudApp().providers.task_processing.report_result(
                task["id"],
                {'speech': speech_id},
            )
        except Exception as e:  # noqa
            print(str(e))
            try:
                log(nc, LogLvl.ERROR, str(e))
                nc.providers.task_processing.report_result(task["id"], None, str(e))
            except:
                pass
            sleep(30)


def start_bg_task():
    t = Thread(target=background_thread_task)
    t.start()


async def enabled_handler(enabled: bool, nc: NextcloudApp) -> str:
    # This will be called each time application is `enabled` or `disabled`
    # NOTE: `user` is unavailable on this step, so all NC API calls that require it will fail as unauthorized.
    if enabled:
        voice_enum_values = [ShapeEnumValue(name=voice_description, value=voice_name) for
                             voice_name, voice_description in VOICE_DESCRIPTIONS.items()]
        await nc.providers.task_processing.register(TaskProcessingProvider(id=TASKPROCESSING_PROVIDER_ID,
                                                                           name="Nextcloud local text to speech",
                                                                           task_type="core:text2speech",
                                                                           optional_input_shape={
                                                                               "voice":
                                                                                   ShapeDescriptor(
                                                                                       name="voice",
                                                                                       description="Voice to use",
                                                                                       shape_type=ShapeType.ENUM),
                                                                               "speed":
                                                                                   ShapeDescriptor(
                                                                                       name="speed",
                                                                                       description="Speech speed modifier",
                                                                                       shape_type=ShapeType.NUMBER),
                                                                           },
                                                                           optional_input_shape_enum_values={
                                                                               "voice": voice_enum_values
                                                                           },
                                                                           input_shape_defaults={
                                                                               "voice": "af_heart",
                                                                               "speed": 1
                                                                           }
                                                                           )
                                                    )
        await nc.log(LogLvl.INFO, f"Hello from {nc.app_cfg.app_name} :)")
        app_enabled.set()
    else:
        await nc.providers.task_processing.unregister(TASKPROCESSING_PROVIDER_ID, True)
        await nc.log(LogLvl.INFO, f"Bye bye from {nc.app_cfg.app_name} :(")
        app_enabled.clear()
    # In case of an error, a non-empty short string should be returned, which will be shown to the NC administrator.
    return ""


if __name__ == "__main__":
    # Creates the storage directory for the models
    # Wrapper around `uvicorn.run`.
    # You are free to call it directly, with just using the `APP_HOST` and `APP_PORT` variables from the environment.
    os.chdir(Path(__file__).parent)
    run_app("main:APP", log_level="trace")
