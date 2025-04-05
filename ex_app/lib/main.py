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
from nc_py_api.ex_app.providers.task_processing import TaskProcessingProvider
from kokoro import KPipeline

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
    pipe = KPipeline(lang_code='a')  # Specifying a language code is done here

    while True:
        if not app_enabled.is_set() or pipe is None:
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

            log(nc, LogLvl.INFO, "generating speech")
            time_start = perf_counter()
            prompt = task.get("input").get('input')

            generator = pipe(prompt, voice='af_heart', speed=1)  # Specifying voice and speed is done here
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
    print(f"enabled={enabled}", flush=True)
    if enabled:
        await nc.providers.task_processing.register(TaskProcessingProvider(id=TASKPROCESSING_PROVIDER_ID,
                                                                           name="Nextcloud local text to speech",
                                                                           task_type="core:text2speech"))
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
