from detectron2.engine import DefaultTrainer

def train_model(cfg):
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    return os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
