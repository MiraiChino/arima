import pathlib

import uvicorn
from domonic.html import *
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import FileResponse, HTMLResponse

import task

app = FastAPI()
task_pred = task.PredictBaken()
task_create = task.CreateBakenHTML()

@app.get("/predict/{race_id}", response_class=HTMLResponse)
def predict_baken(race_id: str, background_tasks: BackgroundTasks):
    task_id = f"/predict/{race_id}"
    doing_task = task.get_doing_task([task_pred, task_create])
    if doing_task and doing_task.get_id() == task_id:
        return str(html(
            body(
                ul(''.join([f'{li(log)}' for log in doing_task.get_logs()])),
            )
        ))
    elif doing_task and doing_task.get_id() != task_id:
        return str(html(body(div(f"Doing {doing_task.get_id()}."))))

    baken_pickle = pathlib.Path(f"{race_id}.predict")
    if baken_pickle.is_file():
        return str(html(body(
            div(f"Already {baken_pickle} exists."),
            div(f"Go /create/{race_id}.")
        )))

    task_pred.init()
    background_tasks.add_task(task_pred, task_id, race_id)
    return str(html(body(
        div(f"Received {task_id}."),
        div("Refresh page to show the status")
    )))

@app.get("/create/{race_id}", response_class=HTMLResponse)
def create_baken_html(race_id: str, background_tasks: BackgroundTasks, top: int=100, odd_th=2.0):
    task_id = f"/create/{race_id}"
    doing_task = task.get_doing_task([task_pred, task_create])
    if doing_task and doing_task.get_id() == task_id:
        return str(html(
            body(
                ul(''.join([f'{li(log)}' for log in doing_task.get_logs()])),
            )
        ))
    elif doing_task and doing_task.get_id() != task_id:
        return str(html(body(div(f"Doing {doing_task.get_id()}."))))
    
    baken_pickle = pathlib.Path(f"{race_id}.predict")
    if not baken_pickle.is_file():
        return str(html(body(
            div(f"Error: Did not find {baken_pickle}."),
            div(f"Go /predict/{race_id} first.")
        )))
    
    baken_html = pathlib.Path(f"{race_id}.html")
    if baken_html.is_file():
        return str(html(body(
            div(f"Already {baken_html} exists."),
            div(f"Go /result/{race_id}.")
        )))
    
    task_create.init()
    background_tasks.add_task(task_create, task_id, race_id, top, odd_th)
    return str(html(body(
        div(f"Received {task_id}."),
        div("Refresh page to show the status")
    )))

@app.get("/update/{race_id}", response_class=HTMLResponse)
def create_baken_html(race_id: str, background_tasks: BackgroundTasks, top: int=100, odd_th=2.0):
    task_id = f"/update/{race_id}"
    doing_task = task.get_doing_task([task_pred, task_create])
    if doing_task and doing_task.get_id() == task_id:
        return str(html(
            body(
                ul(''.join([f'{li(log)}' for log in doing_task.get_logs()])),
            )
        ))
    elif doing_task and doing_task.get_id() != task_id:
        return str(html(body(div(f"Doing {doing_task.get_id()}."))))
    
    baken_pickle = pathlib.Path(f"{race_id}.predict")
    if not baken_pickle.is_file():
        return str(html(body(
            div(f"Error: Did not find {baken_pickle}."),
            div(f"Go /predict/{race_id} first.")
        )))
    
    task_create.init()
    background_tasks.add_task(task_create, task_id, race_id, top, odd_th)
    return str(html(body(
        div(f"Received {task_id}."),
        div("Refresh page to show the status")
    )))

@app.get("/result/{race_id}")
def race_html(race_id: str):
    return FileResponse(f"{race_id}.html")

if __name__ == "__main__":
    uvicorn.run(app, debug=True, host="0.0.0.0", port=8080)
