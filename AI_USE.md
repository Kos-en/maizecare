# Use of AI assistance

Following the project policy, here's where I used AI in this project.

I used Claude Code (by Anthropic), an AI coding assistant, for the following:

- **Project structure.** It advised me on how to lay out the project: which files and folders to have, and how to
  split the code into the data pipeline, the two models, evaluation and the API.
- **Cleaner, more efficient code.** For example, it helped make near-copy detection faster, cache images so photos
  aren't read twice, remove repeated or unused code, and make the code easier to read.
- **Unit tests.** It generated the unit tests in `tests/` to make the project more robust. They check the data
  pipeline, the image cache, preprocessing, both models, the evaluation and the API on every push.
- **CI/CD for deployment.** It generated the GitHub Actions pipeline (`.github/workflows/ci.yml`), which lints and
  tests the code and builds the Docker image on every push, and the Render deployment setup (`render.yaml`).
- **Debugging and documentation.** It helped me track down problems when something broke, and helped with the
  documentation.

The rest is my own work: the problem, the approach, the data, the modelling decisions, running the training and
evaluation, and making sense of the results. I reviewed every suggestion before using it, and I can explain any
part of the code in this repository.
