import torch
import onnx
import onnxruntime
import sys
import os

# Asegúrate de que model.py esté en el path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import LSMTransformer


def infer_hparams(state_dict: dict, cfg: dict) -> dict:
    """
    Infiere los hiperparámetros reales mirando las formas de los tensores
    del state_dict. Evita el error de size mismatch cuando cfg está
    incompleto o usa valores distintos a los defaults.
    """
    def shape(key):
        t = state_dict.get(key)
        return t.shape if t is not None else None

    # embedding.0 es nn.Linear(input_dim, d_model) → weight: (d_model, input_dim)
    emb = shape('embedding.0.weight')
    input_dim = emb[1] if emb else cfg.get('input_dim', 399)
    d_model   = emb[0] if emb else cfg.get('d_model',   512)

    # transformer.layers.0.linear1 → weight: (dim_feedforward, d_model)
    ff = shape('transformer.layers.0.linear1.weight')
    dim_feedforward = ff[0] if ff else cfg.get('dim_feedforward', 2048)

    # glosa_head.net.6 (última Linear) → weight: (num_classes, hidden)
    cls = shape('glosa_head.net.6.weight')
    num_classes = cls[0] if cls else cfg.get('num_classes', 249)

    # trigger_head.net.0 → weight: (hidden, d_model * trigger_window)
    trg0 = shape('trigger_head.net.0.weight')
    trigger_window = (trg0[1] // d_model) if trg0 else cfg.get('trigger_window', 30)

    # Contar capas del transformer
    num_layers = 0
    while f'transformer.layers.{num_layers}.self_attn.in_proj_weight' in state_dict:
        num_layers += 1
    if num_layers == 0:
        num_layers = cfg.get('num_layers', 6)

    hp = dict(
        input_dim       = input_dim,
        d_model         = d_model,
        dim_feedforward = dim_feedforward,
        num_classes     = num_classes,
        trigger_window  = trigger_window,
        num_layers      = num_layers,
        nhead           = cfg.get('nhead',   8),
        dropout         = cfg.get('dropout', 0.1),
        tmax            = cfg.get('tmax',    200),
    )

    print("Hiperparámetros inferidos del checkpoint:")
    for k, v in hp.items():
        print(f"  {k:20s} = {v}")
    print()
    return hp


def export_model_to_onnx(model_path='best_model.ptrom', output_path='model.onnx'):
    print(f"Cargando modelo desde {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu')

    if isinstance(checkpoint, LSMTransformer):
        model = checkpoint

    elif isinstance(checkpoint, dict):
        print(f"Checkpoint detectado. Llaves: {list(checkpoint.keys())}")

        for key in ('model_state', 'state_dict', 'model_state_dict', 'model'):
            if key in checkpoint:
                state_dict = checkpoint[key]
                print(f"State dict extraído desde '{key}'")
                break
        else:
            state_dict = checkpoint

        raw_cfg = checkpoint.get('cfg', checkpoint.get('hparams', checkpoint.get('config', {})))
        cfg = raw_cfg if isinstance(raw_cfg, dict) else {}
        if cfg:
            print(f"cfg encontrado: {cfg}")

        hp = infer_hparams(state_dict, cfg)

        model = LSMTransformer(
            num_classes     = hp['num_classes'],
            input_dim       = hp['input_dim'],
            d_model         = hp['d_model'],
            nhead           = hp['nhead'],
            num_layers      = hp['num_layers'],
            dim_feedforward = hp['dim_feedforward'],
            dropout         = hp['dropout'],
            tmax            = hp['tmax'],
            trigger_window  = hp['trigger_window'],
            use_eadm        = False,
        )

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"⚠️  Pesos faltantes ({len(missing)}): {missing[:5]}")
        if unexpected:
            print(f"⚠️  Pesos inesperados ({len(unexpected)}): {unexpected[:5]}")
        if not missing and not unexpected:
            print("✅ Todos los pesos cargados correctamente")

    else:
        raise TypeError(f"Formato no reconocido: {type(checkpoint)}")

    model.eval()
    print(f"Modelo listo: {model.param_count()}")

    # Entrada dummy con dimensiones reales del modelo cargado
    input_dim = model.embedding[0].in_features
    dummy_x   = torch.randn(1, 210, input_dim, dtype=torch.float32)

    with torch.no_grad():
        glosa_out, trigger_out = model(dummy_x)
    print(f"Forward OK → glosa: {glosa_out.shape}, trigger: {trigger_out.shape}")

    print(f"\nExportando a {output_path} ...")
    torch.onnx.export(
        model,
        dummy_x,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['landmarks'],
        output_names=['glosa_logits', 'trigger_logits'],
        dynamic_axes={
            'landmarks':      {0: 'batch', 1: 'time'},
            'glosa_logits':   {0: 'batch'},
            'trigger_logits': {0: 'batch'},
        },
        verbose=False,
    )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✅ Modelo ONNX válido.")

    ort_session = onnxruntime.InferenceSession(output_path)
    print("\nEntradas:")
    for inp in ort_session.get_inputs():
        print(f"  {inp.name}: {inp.shape}  ({inp.type})")
    print("Salidas:")
    for out in ort_session.get_outputs():
        print(f"  {out.name}: {out.shape}  ({out.type})")

    ort_inputs = {'landmarks': dummy_x.numpy()}
    ort_glosa, ort_trigger = ort_session.run(None, ort_inputs)
    print(f"\nInferencia ONNX OK → glosa: {ort_glosa.shape}, trigger: {ort_trigger.shape}")
    print("\n✅ Todo listo. Archivo generado:", output_path)
    return True


if __name__ == '__main__':
    success = export_model_to_onnx()
    sys.exit(0 if success else 1)