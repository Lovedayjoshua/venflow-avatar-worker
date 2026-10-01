# Venflow Avatar Worker

Production GPU worker for the Venflow AI Clone platform.

The worker is designed for RunPod Serverless and uses MuseTalk 1.5 for audio-driven avatar video generation.

## Architecture

Venflow API
    |
    v
RunPod Serverless
    |
    v
Venflow Avatar Worker
    |
    v
MuseTalk 1.5
    |
    v
Generated Avatar Video

## Input

The worker accepts:

- source video or image URL
- audio URL

Example:

```json
{
  "input": {
    "video_url": "https://example.com/avatar.mp4",
    "audio_url": "https://example.com/response.wav"
  }
}