from detectron2.engine import DefaultPredictor

def setup_predictor(cfg, model_weights_path, score_threshold=0.7):
    cfg.MODEL.WEIGHTS = model_weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
    return DefaultPredictor(cfg)
