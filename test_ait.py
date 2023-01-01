from nataili.aitemplate import AITemplate
from nataili import ModelManager, logger

mm = ModelManager()

mm.aitemplate.load()


def run():
    while True:
        logger.info("init")
        ait = AITemplate(mm.aitemplate.loaded_models["ait"])
        logger.info("start")
        ait.generate("corgi", ddim_steps=30)
        logger.info("end")


if __name__ == "__main__":
    run()
