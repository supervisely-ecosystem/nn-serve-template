import supervisely_lib as sly

my_app = sly.AppService()


@my_app.callback("get_output_classes_and_tags")
@sly.timeit
def merge(api: sly.Api, task_id, context, state, app_logger):
    app_logger.debug("Input context", extra={"context": context})
    app_logger.debug("Input data", extra={"state": state})

    request_id = context["request_id"]

    classes = sly.ObjClassCollection([
        sly.ObjClass("person", sly.Rectangle),
        sly.ObjClass("car", sly.Polygon),
        sly.ObjClass("dog", sly.Bitmap),
    ])
    tags = sly.TagMetaCollection([
        sly.TagMeta("confidence", sly.TagValueType.ANY_NUMBER)
    ])
    meta = sly.ProjectMeta(classes, tags)
    print(meta)
    app_logger.debug("Result project meta (classes + tags)", extra={"meta": meta.to_json()})

    my_app.send_response(request_id, data=meta.to_json())


def main():
    sly.logger.info("Script arguments", extra={})
    my_app.run()


if __name__ == "__main__":
    sly.main_wrapper("main", main)
