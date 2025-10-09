import os
import numpy as np

COLUMNS = {
    "device": 0,
    "accel_x": 1, "accel_y": 2, "accel_z": 3,
    "gyro_x": 4,  "gyro_y": 5,  "gyro_z": 6,
    "mag_x": 7,   "mag_y": 8,   "mag_z": 9,
    "timestamp": 10, "label": 11,
}

def _load_csv(path):
    """Carrega um CSV garantindo que o resultado é sempre 2D float."""
    kwargs = {"delimiter": ",", "dtype": float, "ndmin": 2}
    # Leitura robusta ignorando a primeira linha de cabeçalho se existir
    try:
        data = np.genfromtxt(path, skip_header=1, **kwargs)
        if data.size == 0 or np.isnan(data).all():
            # Tentar sem skip_header
            data = np.genfromtxt(path, skip_header=0, **kwargs)
    except Exception:
        data = np.genfromtxt(path, skip_header=0, **kwargs)

    data = np.asarray(data, dtype=float)

    # Normalizar dimensão para 2D, garantindo 12 colunas, mesmo quando o ficheiro está vazio
    if data.ndim == 1:
        if data.size == 0:
            data = np.empty((0, len(COLUMNS)), dtype=float)
        else:
            data = data.reshape(1, -1)

    if data.shape[0] == 0:
        return np.empty((0, len(COLUMNS)), dtype=float)

    # Remover linhas que não contenham dados válidos
    finite_mask = np.isfinite(data).any(axis=1)
    data = data[finite_mask]

    if data.size == 0:
        return np.empty((0, len(COLUMNS)), dtype=float)

    if data.shape[1] < len(COLUMNS):
        raise ValueError(
            f"CSV em {path} possui apenas {data.shape[1]} colunas, mas são esperadas {len(COLUMNS)}"
        )

    return data[:, :len(COLUMNS)]

def load_participant(data_root,participant_id):
    """
    Carrega todos os dispositivos (1..5) do participante 'participant_id'.
    Devolve um único array numpy [N, 12] concatenado por tempo/linhas.
    """
    part_dir = os.path.join(data_root, f"part{participant_id}")
    arrays = []
    for dev in range(1, 6):
        csv_path = os.path.join(part_dir, f"part{participant_id}dev{dev}.csv")
        if os.path.exists(csv_path):
            arr = _load_csv(csv_path)
            if arr.size == 0:
                continue
            arrays.append(arr)
    if not arrays:
        raise FileNotFoundError(f"Nenhum CSV encontrado para participante {participant_id} em {part_dir}")
    return np.vstack(arrays)

def load_all(data_root, participants=None):
    """
    Carrega todos os participantes especificados (ou deteta todos partX/).
    Retorna um único array [N, 12].
    """
    if participants is None:
        participants = []
        for name in sorted(os.listdir(data_root)):
            if name.startswith("part"):
                try:
                    pid = int(name.replace("part", ""))
                    participants.append(pid)
                except:
                    pass
    if not participants:
        raise FileNotFoundError(f"Nenhum participante encontrado em {data_root}")

    arrays = []
    missing = []
    for pid in participants:
        try:
            arr = load_participant(data_root, pid)
        except FileNotFoundError:
            missing.append(pid)
            continue
        if arr.size:
            arrays.append(arr)
        else:
            missing.append(pid)
    if not arrays:
        if missing:
            raise FileNotFoundError(
                "Nenhum CSV válido encontrado para os participantes: "
                + ", ".join(str(pid) for pid in missing)
            )
        raise ValueError("Nenhum dado disponível para concatenação")
    return np.vstack(arrays)

