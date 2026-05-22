import logging
from dotenv import load_dotenv
import dagshub

load_dotenv()

dagshub.init(repo_owner='RodrigoSchneiderbr', repo_name='MLOPS', mlflow=True)

# Configuração do log
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler()
    ]
)
