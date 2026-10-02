import runpod


def handler(job):
    job_input = job["input"]

    return {
        "status": "success",
        "message": "Venflow Avatar Worker is running",
        "input": job_input,
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})