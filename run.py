from cytomine.models import ImageInstanceCollection, AnnotationCollection, Annotation, Job, AttachedFileCollection
from shapely.affinity import affine_transform
from sldc.locator import mask_to_objects_2d
from cytomine import CytomineJob
from pathlib import Path
from PIL import Image

import joblib
import torch
import utils
import sys
import os


def main(argv):
    with CytomineJob.from_cli(argv) as cj:
        cj.job.update(status=Job.RUNNING, progress=0, status_comment="Initialization...")
        DIR = str(Path.home())
        dataset_path = os.path.join(DIR, "dataset_to_predict")
        images_path = os.path.join(dataset_path, "images")
        model_path = os.path.join(DIR, "models", str(cj.parameters.cytomine_id_job))

        # Fetching parameters from the training job
        training_job = Job().fetch(cj.parameters.cytomine_id_job)
        attached_files = AttachedFileCollection(training_job).fetch()
        parameters_file = attached_files.find_by_attribute("filename", "parameters")
        parameters_filepath = os.path.join(DIR, "parameters")
        parameters_file.download(parameters_filepath, override=True)

        training_parameters = joblib.load(parameters_filepath)

        # Creating directories
        mask_path = os.path.join(dataset_path, "predicted_"+training_parameters["cytomine_term_name"])
        os.makedirs(dataset_path, exist_ok=True)
        os.makedirs(images_path, exist_ok=True)
        os.makedirs(mask_path, exist_ok=True)
        os.makedirs(model_path, exist_ok=True)

        # Fetching model
        cj.job.update(progress=10, status_comment="Fetching the trained model...")
        model_file = attached_files.find_by_attribute("filename", "model.pth")
        model_filepath = os.path.join(model_path, "model.pth")
        model_file.download(model_filepath, override=True)
        model = utils.load_model(model_filepath)

        device_name = "cpu"
        if torch.cuda.is_available():
            device_name = 'cuda:0'
        DEVICE = torch.device(device_name)

        model.to(DEVICE)
        model.eval()

        # Fetching the IDs of the images to predict
        images = ImageInstanceCollection().fetch_with_filter("project", cj.parameters.cytomine_id_project)
        images_id = []
        if cj.parameters.images_to_predict == "all":
            images_id = [image.id for image in images]
        else:
            images_id = [int(im_id) for im_id in cj.parameters.images_to_predict.split(',')]

        # Prediting segmentation masks
        cj.job.update(progress=30, status_comment="Making predictions...")
        annotations = AnnotationCollection()
        for image in images:
            if image.id in images_id:
                image.download(os.path.join(images_path, image.originalFilename))
                pil_image = Image.open(os.path.join(images_path, image.originalFilename))
                w, h = pil_image.size

                # Transforming image to a square image 512 by 512 pixels and feeding it to the nework
                tensor_image, untransform = utils.transform_image(pil_image, size=512)
                mask = utils.create_segmentation_mask(model, tensor_image, untransform, DEVICE, cj.parameters.threshold)
                slices = mask_to_objects_2d(mask)

                # Creating annotations
                annotations.extend([
                    Annotation(location=affine_transform(poly, [1, 0, 0, -1, 0, h]).wkt,
                               id_image=image.id,
                               id_terms=[training_parameters["cytomine_term_id"]],
                               id_project=cj.parameters.cytomine_id_project)
                    for poly, _ in slices
                ])

        # Upload annotations
        cj.job.update(progress=70, statusComment="Uploading extracted annotation...")
        annotations.save()

        # Finished
        cj.job.update(status=Job.TERMINATED, progress=100, statusComment="Finished.")


if __name__ == '__main__':
    main(sys.argv[1:])
