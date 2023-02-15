from pathlib import Path
from socket import gethostname

MAX_THREADS = 4

hostname = gethostname()

if 'dabler-ThinkPad-P1-Gen-2' in hostname:
    p_base = Path('/media/dabler/ext1TB/instrumented-slippers/')
elif 'lxhultrafast' in hostname:
    p_base = Path('/home/daniel/instrumented-slippers/')

p_raw_data = p_base.joinpath('raw-data')
p_processing_results = p_base.joinpath('processing-results')
p_processing_results.mkdir(exist_ok=True, parents=True)

p_repo = Path(__file__).parent
p_logger_config = p_repo.joinpath('logging.ini')

