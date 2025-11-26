"""Kokoro Text2Speech App

Module is part of the Nextcloud Kokoro Text2Speech App. It contains the main functionality for converting text to
speech using various voices and languages. The app integrates with Nextcloud, allowing task processing for
text-to-speech conversion, and utilizes libraries such as soundfile and torch for audio processing. The application is
built to run within a Docker container and can be managed and executed through specific makefile targets. The main
components include background task processing, Nextcloud application lifecycle management, and voice generation.
"""

import asyncio
import io
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from json import JSONDecodeError
from pathlib import Path
from threading import Event, Thread
from time import perf_counter, sleep
import traceback
from audiometa import update_metadata, UnifiedMetadataKey
import static_ffmpeg
static_ffmpeg.add_paths()

import niquests
import soundfile as sf
import torch
from fastapi import FastAPI
from kokoro import KPipeline, KModel
from nc_py_api import NextcloudApp, NextcloudException
from nc_py_api.ex_app import AppAPIAuthMiddleware, LogLvl, run_app, set_handlers, get_computation_device, \
    persistent_storage
from nc_py_api.ex_app.providers.task_processing import (
    ShapeDescriptor,
    ShapeEnumValue,
    ShapeType,
    TaskProcessingProvider, TaskType,
)

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
    # Translated with Google Translate
    "jf_alpha": "Alpha (日本語 女性)",  # Japanese Female
    "jf_gongitsune": "Gongitsune (日本語 女性)",  # Japanese Female
    "jf_nezumi": "Nezumi (日本語 女性)",  # Japanese Female
    "jf_tebukuro": "Tebukuro (日本語 女性)",  # Japanese Female
    "jm_kumo": "Kumo (日本語 男性)",  # Japanese Male
    "zf_xiaobei": "Xiaobei (普通话 女性)",  # Mandarin Chinese Female
    "zf_xiaoni": "Xiaoni (普通话 女性)",  # Mandarin Chinese Female
    "zf_xiaoxiao": "Xiaoxiao (普通话 女性)",  # Mandarin Chinese Female
    "zf_xiaoyi": "Xiaoyi (普通话 女性)",  # Mandarin Chinese Female
    "zm_yunjian": "Yunjian (普通话 男性)",  # Mandarin Chinese Male
    "zm_yunxi": "Yunxi (普通话 男性)",  # Mandarin Chinese Male
    "zm_yunxia": "Yunxia (普通话 男性)",  # Mandarin Chinese Male
    "zm_yunyang": "Yunyang (普通话 男性)",  # Mandarin Chinese Male
    "ef_dora": "Dora (Español Femenino)",  # Spanish Female
    "em_alex": "Alex (Español Masculino)",  # Spanish Male
    "em_santa": "Santa (Español Masculino)",  # Spanish Male
    "ff_siwis": "Siwis (Français Féminin)",  # French Female
    "hf_alpha": "Alpha (हिन्दी महिला)",  # Hindi Female
    "hf_beta": "Beta (हिन्दी महिला)",  # Hindi Female
    "hm_omega": "Omega (हिन्दी पुरुष)",  # Hindi Male
    "hm_psi": "Psi (हिन्दी पुरुष)",  # Hindi Male
    "if_sara": "Sara (Italiano Femminile)",  # Italian Female
    "im_nicola": "Nicola (Italiano Maschile)",  # Italian Male
    "pf_dora": "Dora (Português Brasileiro Feminino)",  # Brazilian Portuguese Female
    "pm_alex": "Alex (Português Brasileiro Masculino)",  # Brazilian Portuguese Male
    "pm_santa": "Santa (Português Brasileiro Masculino)",  # Brazilian Portuguese Male
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def log(nc, level, content):
    logger.log((level + 1) * 10, content)
    if level < LogLvl.WARNING:
        return
    try:
        asyncio.run(nc.log(level, content))
    except Exception:
        logger.exception("Failed to log to Nextcloud")


TASKPROCESSING_PROVIDER_ID = "text2speech_kokoro"
TASKPROCESSING_TYPE = "core:text2speech"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global TASKPROCESSING_TYPE
    set_handlers(
        APP,
        enabled_handler,
        trigger_handler=trigger_handler,
        models_to_fetch=models_to_fetch
    )
    nc = NextcloudApp()
    if nc.srv_version.get("major") < 32:
        TASKPROCESSING_TYPE = "kokoro:text2speech"
    if nc.enabled_state:
        app_enabled.set()
    start_bg_task()
    yield


APP = FastAPI(lifespan=lifespan)
APP.add_middleware(AppAPIAuthMiddleware)  # set global AppAPI authentication middleware

app_enabled = Event()
TRIGGER = Event()
WAIT_INTERVAL = 5
WAIT_INTERVAL_WITH_TRIGGER = 5 * 60

REPO_ID = "Nextcloud-AI/Kokoro-82M"

models_to_fetch = {
    "https://huggingface.co/Nextcloud-AI/Kokoro-82M/resolve/main/kokoro-v1_0.pth": {
        "save_path": os.path.join(persistent_storage(), "kokoro-v1_0.pth")},
}


def background_thread_task():
    nc = NextcloudApp()
    while not app_enabled.is_set():
        sleep(5)
    print("Starting background task")
    pipes = {}  # Stores for each model a KPipeline instance
    model = KModel(repo_id=REPO_ID, model=os.path.join(persistent_storage(), "kokoro-v1_0.pth"))

    while True:
        if not app_enabled.is_set():
            sleep(30)
            continue
        try:
            next_task = nc.providers.task_processing.next_task([TASKPROCESSING_PROVIDER_ID], [TASKPROCESSING_TYPE])
            if "task" not in next_task or next_task is None:
                wait_for_task()
                continue
            task = next_task.get("task")
        except (NextcloudException, JSONDecodeError) as e:
            tb_str = "".join(traceback.format_exception(e))
            log(nc, LogLvl.WARNING, f"Error fetching the next task {tb_str}")
            wait_for_task(10)
            continue
        except (
                niquests.HTTPError
        ) as e:
            tb_str = "".join(traceback.format_exception(e))
            log(nc, LogLvl.DEBUG, f"Ignored error during task polling {tb_str}")
            wait_for_task(2)
            continue
        pipes = handle_task(nc, task, pipes, model)


def handle_task(nc, task, pipes, model):
    try:
        log(nc, LogLvl.INFO, f"Next task: {task['id']}")
        prompt = task.get("input").get("input")
        voice = task.get("input").get("voice") or "af_heart"  # Use 'af_heart' if voice is not specified
        lang_code = voice[0]
        speed = task.get("input").get("speed") or 1

        log(nc, LogLvl.INFO, "generating speech with voice: " + voice + " and speed: " + str(speed))
        time_start = perf_counter()
        pipe = pipes.get(lang_code)
        if pipe is None:
            device = get_computation_device().lower()
            if device not in ("cpu", "cuda"):
                device = "cpu"
            pipe = KPipeline(lang_code=lang_code, device=device, repo_id=REPO_ID, model=model)
            pipes[lang_code] = pipe
        speechs = []
        for _, _, speech in pipe(prompt, voice=voice, speed=speed):
            speechs.append(speech)
        speech = torch.cat(speechs, dim=0)
        log(nc, LogLvl.INFO, f"speech generated: {perf_counter() - time_start}s")

        # export tensors to wave
        speech_stream = io.BytesIO()
        speech_stream.name = "speech.wav"
        sf.write(speech_stream, speech, 24000)
        speech_stream.seek(0)

        # add metadata to wave
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_file:
            temp_filename = tmp_file.name
            tmp_file.write(speech_stream.getvalue())
            new_metadata = {
                UnifiedMetadataKey.COMMENT: 'Generated using Artificial Intelligence',
            }
            update_metadata(temp_filename, new_metadata)

            # Read the modified file back into a BytesIO stream
            output_stream = io.BytesIO()
            with open(temp_filename, "rb") as f:
                output_stream.write(f.read())
            output_stream.seek(0)

        try:
            speech_id = nc.providers.task_processing.upload_result_file(task.get("id"), output_stream)
        except Exception:
            speech_id = nc.providers.task_processing.upload_result_file(task.get("id"), output_stream)
        try:
            NextcloudApp().providers.task_processing.report_result(
                task["id"],
                {"speech": speech_id},
            )
        except Exception:
            NextcloudApp().providers.task_processing.report_result(
                task["id"],
                {"speech": speech_id},
            )
    except Exception as e:  # noqa
        print(str(e))
        try:
            log(nc, LogLvl.ERROR, str(e))
            nc.providers.task_processing.report_result(task["id"], None, str(e))
        except Exception:
            log(nc, LogLvl.ERROR, "Failed to report error in task result")
    return pipes


def start_bg_task():
    t = Thread(target=background_thread_task)
    t.start()


async def enabled_handler(enabled: bool, nc: NextcloudApp) -> str:
    global TASKPROCESSING_TYPE
    # This will be called each time application is `enabled` or `disabled`
    # NOTE: `user` is unavailable on this step, so all NC API calls that require it will fail as unauthorized.
    if enabled:
        voice_enum_values = [
            ShapeEnumValue(name=voice_description, value=voice_name)
            for voice_name, voice_description in VOICE_DESCRIPTIONS.items()
        ]
        new_task_type = None
        TASKPROCESSING_TYPE = "core:text2speech"
        # Check if Nextcloud version is less than 32 to create a custom task type when needed
        if (await nc.srv_version).get("major") < 32:
            await nc.log(LogLvl.INFO, f"Creating custom task type for {nc.app_cfg.app_name}")
            new_task_type = TaskType(
                id="kokoro:text2speech",
                name="Text to speech",
                description="Text to speech",
                input_shape=[
                    ShapeDescriptor(name="input", description="Prompt", shape_type=ShapeType.TEXT),
                ],
                output_shape=[
                    ShapeDescriptor(name="speech", description="Output speech", shape_type=ShapeType.AUDIO),
                ]
            )
            TASKPROCESSING_TYPE = "kokoro:text2speech"
        await nc.providers.task_processing.register(
            TaskProcessingProvider(
                id=TASKPROCESSING_PROVIDER_ID,
                name="Kokoro local text to speech",
                task_type=TASKPROCESSING_TYPE,
                optional_input_shape=[
                    ShapeDescriptor(name="voice", description="Voice to use", shape_type=ShapeType.ENUM),
                    ShapeDescriptor(name="speed", description="Speech speed modifier", shape_type=ShapeType.NUMBER),
                ],
                optional_input_shape_enum_values={"voice": voice_enum_values},
                input_shape_defaults={"voice": "af_heart", "speed": 1},
            ),
            new_task_type
        )
        await nc.log(LogLvl.INFO, f"Hello from {nc.app_cfg.app_name} :)")
        app_enabled.set()
    else:
        await nc.providers.task_processing.unregister(TASKPROCESSING_PROVIDER_ID, True)
        await nc.log(LogLvl.INFO, f"Bye bye from {nc.app_cfg.app_name} :(")
        app_enabled.clear()
    # In case of an error, a non-empty short string should be returned, which will be shown to the NC administrator.
    return ""


# Will only be called on Nextcloud 33+
def trigger_handler(_: str):
    TRIGGER.set()


# Wait for `interval` seconds or `WAIT_INTERVAL` if `interval` is not set
# If `TRIGGER` gets set in the meantime, we override `WAIT_INTERVAL` with WAIT_INTERVAL_WITH_TRIGGER
def wait_for_task(interval=None):
    global WAIT_INTERVAL
    if interval is None:
        interval = WAIT_INTERVAL
    if TRIGGER.wait(timeout=interval):
        WAIT_INTERVAL = WAIT_INTERVAL_WITH_TRIGGER
    TRIGGER.clear()


if __name__ == "__main__":
    # Creates the storage directory for the models
    # Wrapper around `uvicorn.run`.
    # You are free to call it directly, with just using the `APP_HOST` and `APP_PORT` variables from the environment.
    os.chdir(Path(__file__).parent)
    run_app("main:APP", log_level="trace")
