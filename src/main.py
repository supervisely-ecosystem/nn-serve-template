import math
import numpy as np
import cv2
import supervisely_lib as sly

my_app = sly.AppService()
meta: sly.ProjectMeta = None


def init_output_meta():
    global meta
    classes = sly.ObjClassCollection([
        sly.ObjClass("person", sly.Rectangle),
        sly.ObjClass("car", sly.Polygon),
        sly.ObjClass("dog", sly.Bitmap),
    ])
    tags = sly.TagMetaCollection([
        sly.TagMeta("confidence", sly.TagValueType.ANY_NUMBER)
    ])
    meta = sly.ProjectMeta(classes, tags)


def inference(image: np.ndarray, debug_visualization=False) -> sly.Annotation:
    height, width = image.shape[:2]
    ann = sly.Annotation(img_size=(width, height))

    # generate bbox
    rect = sly.Rectangle(top=0, left=0, bottom=math.floor(height / 2), right=math.floor(width / 2))
    tag = sly.Tag(meta.get_tag_meta("confidence"), 0.75)
    label1 = sly.Label(rect, meta.get_obj_class("person"), sly.TagCollection([tag]))
    ann = ann.add_label(label1)

    # generate polygon
    poly = sly.Polygon(exterior=[sly.PointLocation(math.floor(height / 4), math.floor(width / 2) + 3),
                                 sly.PointLocation(3, math.floor(width / 4 * 3)),
                                 sly.PointLocation(math.floor(height / 4), math.floor(width / 4 * 3)),
                                 sly.PointLocation(math.floor(height / 4), math.floor(width - 3)),
                                 sly.PointLocation(math.floor(height / 2), math.floor(width / 4 * 3))],
                       interior=[])
    label2 = sly.Label(poly, meta.get_obj_class("car"))
    ann = ann.add_label(label2)

    # generate mask
    np_mask = np.zeros((math.floor(width / 2), math.floor(height / 2)), np.uint8)
    mask_h, mask_w = np_mask.shape[:2]
    cv2.circle(np_mask,
               center=(math.floor(mask_w / 4), math.floor(mask_h / 2)),
               radius=math.floor(mask_w / 5),
               color=(1),
               thickness=-1)
    mask = sly.Bitmap(np_mask.astype(bool), origin=sly.PointLocation(math.floor(height / 2), math.floor(width / 2)))
    label3 = sly.Label(mask, meta.get_obj_class("dog"))
    ann = ann.add_label(label3)

    if debug_visualization is True:
        # visualize for debug purposes
        vis_filled = np.zeros((height, width, 3), np.uint8)
        ann.draw(vis_filled)
        vis = cv2.addWeighted(image, 1, vis_filled, 0.5, 0)
        ann.draw_contour(vis, thickness=5)
        sly.image.write("vis.jpg", vis)

    return ann.to_json()


@my_app.callback("get_output_classes_and_tags")
@sly.timeit
def merge(api: sly.Api, task_id, context, state, app_logger):
    request_id = context["request_id"]
    my_app.send_response(request_id, data=meta.to_json())


@my_app.callback("inference_image_id")
@sly.timeit
def inference_image_id(api: sly.Api, task_id, context, state, app_logger):
    app_logger.debug("Input data", extra={"state": state})
    image_id = state["image_id"]
    debug_visualization = state.get("debug_visualization", False)
    image = api.image.download_np(image_id)  # RGB image
    ann = inference(image, debug_visualization)

    request_id = context["request_id"]
    my_app.send_response(request_id, data=ann.to_json())


def debug_inference():
    image = sly.image.read("demo.jpeg")
    ann = _inference(image)
    #print(json.dumps(ann, indent=4))


#@TODO:
#inference_###_batch
#inference_image_part
#inference_image_url
#inference_video_frame
#inference_image_sliding_window
#inference_batch
def main():
    sly.logger.info("Script arguments", extra={})
    init_output_meta()
    #debug_inference()
    my_app.run()


if __name__ == "__main__":
    sly.main_wrapper("main", main)
