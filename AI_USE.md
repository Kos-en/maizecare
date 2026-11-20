# Use of AI assistance

In line with the course policy, this is where I used AI in this project.

**Tool:** Claude Code (Anthropic), an AI coding assistant.

**What it was used for:**

- **Project structure.** Advice on how to arrange the project: which files and folders to have, and how to split
  the code into the data pipeline, the two models, evaluation and the API.
- **Making the code more efficient and cleaner.** For example, faster near-copy detection, caching images so
  photos are not re-read, removing repeated or unused code, and making the code easier to read.
- **Unit tests.** The AI generated the unit tests in `tests/`, to make the project more robust: they check the
  data pipeline, the image cache, preprocessing, both models, the evaluation and the API on every push.
- **CI/CD for deployment.** The AI generated the GitHub Actions pipeline (`.github/workflows/ci.yml`) that lints
  and tests the code and builds the Docker image on every push, and the Render deployment setup (`render.yaml`).
- **Debugging help** when something broke, and help with documentation.

**What I did:** the problem, the approach, the data, the modelling decisions, running the training and
evaluation, and interpreting the results are my own work. I reviewed every suggestion before using it, and I can
explain every part of the code in this repository.
