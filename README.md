## HyperBinder on BEAM 10M

We evaluated HyperBinder on the BEAM 10M benchmark. HyperBinder achieved **96% accuracy** — the highest published score on BEAM 10M to date.

### Reproduce our results:

```bash
# Clone the repository
git clone https://github.com/SemanticReach/BEAM.git
cd BEAM

# Install dependencies
pip install -r requirements.txt

# Set up environment variables in .env file:
# HB_API_KEY=your_hyperbinder_api_key
# OPENAI_API_KEY=your_openai_api_key
# HB_SERVER_URL=http://your-hyperbinder-server:8000

# Run full evaluation (ingests + evaluates, no manual namespace needed)
python beam_ingest.py --all-chats chats/10M --size 10M
