import json
import supervisely_lib as sly

api = sly.Api.from_env()

d = {
    "a": 1,
    "b": 2
}

resp = api.task.send_request(task_id=2125, method="get_output_classes_and_tags", data=d)
print(json.dumps(resp, indent=4))

meta = sly.ProjectMeta.from_json(resp)
print(meta)
