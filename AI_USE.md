<!-- DRAFT: check that every line below is true for you, change what is not, then delete this comment. -->

# Use of AI assistance

In line with the course policy, this is an honest account of where I used AI tools in this project, and where I
did not.

## Tool used

- **Claude Code** (Anthropic), an AI coding assistant that runs in the terminal and works on the repository
  directly.

## What I used it for

- **Code structure and clean-up.** Turning my experiment code into the `maize` package (`src/maize/`), splitting it
  into the data pipeline, the two models, evaluation and the API, and removing dead or repeated code.
- **Engineering around the models.** The FastAPI service, the Dockerfile and docker-compose, the GitHub Actions CI,
  dependency pinning with `uv`, and the TypeScript frontend.
- **Tests.** Writing and fixing the pytest suite in `tests/`.
- **Debugging.** For example, near-copy detection chaining unrelated photos together when only one hash was
  used, and damaged cache files on Google Drive.
- **The results notebook and the documentation.** Drafting the notebook's plots and the README, ARCHITECTURE and
  REPORT files, which I then reviewed and edited.

## What I did myself

- Chose the problem and the approach: a disease classifier plus a healthy-only autoencoder for severity, as set
  out in my proposal.
- Collected and organised the dataset, and ran every data-preparation, training and evaluation job on Google
  Colab.
- Made the modelling decisions: the backbones to compare, the targets (a)-(c), the leaf mask approach, and
  choosing models on validation and touching the test set only once.
- Read and interpreted the results, and wrote the conclusions and limitations.

## How I checked the AI's work

- I read the code it produced before keeping it, and asked for changes when it did not match what I wanted.
- Every module has automated tests, run locally and in CI on every push.
- All reported numbers come from my own runs of the code on my data (see the results notebook). None were
  written by the AI.
- I am responsible for everything submitted here and can explain any part of it.
