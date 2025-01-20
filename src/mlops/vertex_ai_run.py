from google.cloud import aiplatform

# Indstil projekt og region
aiplatform.init(
    project="your-project-id",
    location="your-region",  # fx "us-central1"
)

# Indlæs data og træningsspecifikationer
job = aiplatform.CustomTrainingJob(
    display_name="mlops-training-job",
    script_path="train.py",  # Din træningsscript
    container_uri="us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-3:latest",  # Brug en passende container
    requirements=["torch", "google-cloud-storage"],
)

# Kør træningsjobbet
model = job.run(
    dataset=None,  # Du kan integrere Vertex AI-datasets, hvis nødvendigt
    base_output_dir="gs://your-bucket-name/training-output",
    args=[
        "--bucket-name", "mlops-bucket-224229-1",
        "--image-folder", "raw/img_align_celeba/img_align_celeba",
        "--labels-path", "raw/list_attr_celeba.csv",
    ],
)
