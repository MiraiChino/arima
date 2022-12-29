import asyncio
import pathlib

import uvicorn
from domonic.html import *
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from sse_starlette.sse import EventSourceResponse

import task

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
task_pred = task.PredictBaken()
task_create = task.CreateBakenHTML()
streamlog_html = str(html(
    head(
        script(f"""
            const logserver_url = `${{location.protocol}}//${{location.host}}/log`
            const source = new EventSource(logserver_url);
            source.onmessage = (event) => {{
                document.getElementById("log").innerHTML += event.data + "<br>";
                if (event.data.includes("{task.DONE}")) {{
                    console.log("Closing connection.")
                    source.close()
                }}
            }};
        """)
    ),
    body(
        div(_id="log"),
    )
))

@app.get('/log')
async def stream_log(request: Request):
    async def log_generator(req):
        doing_task = task.get_doing_task([task_pred, task_create])
        i_yielded = 0
        while doing_task and doing_task.is_doing():
            disconnected = await req.is_disconnected()
            if disconnected:
                print(f"Disconnecting client {req.client}")
                break
            await asyncio.sleep(1.0)
            logs = doing_task.get_logs()
            if i_yielded < len(logs):
                for log in logs[i_yielded:]:
                    yield log
                i_yielded = len(logs)
    return EventSourceResponse(log_generator(request))

@app.get("/", response_class=HTMLResponse)
def start_page():
    return str(html(body(
        span("race_id: "),
        input(type="number", _id="input"),
        br(),
        a(_id="link"),
        script("""
            const input = document.getElementById('input');
            const link = document.getElementById('link');

            input.oninput = (e) => {
                const id = e.target.value;
                link.textContent = `Go /predict/${id}`;
                link.href = `${location.protocol}//${location.host}/predict/${id}`;
            };
        """)
    )))

@app.get("/predict/{race_id}", response_class=HTMLResponse)
def predict_baken(race_id: str, request: Request, background_tasks: BackgroundTasks):
    task_id = f"/predict/{race_id}"
    doing_task = task.get_doing_task([task_pred, task_create])
    if doing_task and doing_task.get_id() == task_id:
        return streamlog_html
    elif doing_task and doing_task.get_id() == task.NO_ID:
        pass
    elif doing_task and doing_task.get_id() != task_id:
        return str(html(body(a(f"Doing {doing_task.get_id()}.", _href=f"{request.base_url}{doing_task.get_id()[1:]}"))))

    baken_pickle = pathlib.Path(f"{race_id}.predict")
    if baken_pickle.is_file():
        return str(html(body(
            div(f"Already {baken_pickle} exists."),
            a(f"Go /create/{race_id}", _href=f"{request.base_url}create/{race_id}"),
        )))

    task_pred.init()
    background_tasks.add_task(task_pred, task_id, race_id, next_url=f"{request.base_url}create/{race_id}")
    return streamlog_html

@app.get("/create/{race_id}", response_class=HTMLResponse)
def create_baken_html(race_id: str, request: Request, background_tasks: BackgroundTasks, top: int=100, odd_th: float=2.0):
    task_id = f"/create/{race_id}"
    doing_task = task.get_doing_task([task_pred, task_create])
    if doing_task and doing_task.get_id() == task_id:
        return streamlog_html
    elif doing_task and doing_task.get_id() == task.NO_ID:
        pass
    elif doing_task and doing_task.get_id() != task_id:
        return str(html(body(a(f"Doing {doing_task.get_id()}.", _href=f"{request.base_url}{doing_task.get_id()[1:]}"))))
    
    baken_pickle = pathlib.Path(f"{race_id}.predict")
    if not baken_pickle.is_file():
        return str(html(body(
            div(f"Not found {baken_pickle}."),
            a(f"Go /predict/{race_id}", _href=f"{request.base_url}predict/{race_id}"),
        )))
    
    baken_html = pathlib.Path(f"{race_id}.html")
    if baken_html.is_file():
        return str(html(body(
            div(f"Already {baken_html} exists."),
            a(f"Go /result/{race_id}", _href=f"{request.base_url}result/{race_id}"),
        )))
    
    task_create.init()
    background_tasks.add_task(task_create, task_id, race_id, top, odd_th, next_url=f"{request.base_url}result/{race_id}")
    return streamlog_html

@app.get("/update/{race_id}", response_class=HTMLResponse)
def create_baken_html(race_id: str, request: Request, background_tasks: BackgroundTasks, top: int=100, odd_th: float=2.0):
    task_id = f"/update/{race_id}"
    task_create.init()
    doing_task = task.get_doing_task([task_pred, task_create])
    if doing_task and doing_task.get_id() == task_id:
        return streamlog_html
    elif doing_task and doing_task.get_id() == task.NO_ID:
        pass
    elif doing_task and doing_task.get_id() != task_id:
        return str(html(body(a(f"Doing {doing_task.get_id()}.", _href=f"{request.base_url}{doing_task.get_id()[1:]}"))))
    
    baken_pickle = pathlib.Path(f"{race_id}.predict")
    if not baken_pickle.is_file():
        return str(html(body(
            div(f"Not found {baken_pickle}."),
            a(f"Go /predict/{race_id}", _href=f"{request.base_url}predict/{race_id}"),
        )))
    
    print("dow update")
    task_create.init()
    background_tasks.add_task(task_create, task_id, race_id, top, odd_th, next_url=f"{request.base_url}result/{race_id}")
    return streamlog_html

@app.get("/result/{race_id}")
def race_html(race_id: str, request: Request):
    baken_html = pathlib.Path(f"{race_id}.html")
    if baken_html.is_file():
        return FileResponse(baken_html)
    else:
        return HTMLResponse(str(html(body(
            div(f"Not found {baken_html}."),
            a(f"Go /create/{race_id}", _href=f"{request.base_url}create/{race_id}"),
        ))))

if __name__ == "__main__":
    uvicorn.run(app, debug=True, host="0.0.0.0", port=8080)
