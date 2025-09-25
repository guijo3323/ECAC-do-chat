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
    # Leitura robusta ignorando a primeira linha de cabeçalho se existir
    try:
        data = np.genfromtxt(path, delimiter=",", skip_header=1)
        if np.isnan(data).all():
            # Tentar sem skip_header
            data = np.genfromtxt(path, delimiter=",", skip_header=0)
    except Exception:
        data = np.genfromtxt(path, delimiter=",", skip_header=0)
    return data

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
    arrays = [load_participant(data_root, pid) for pid in participants]
    return np.vstack(arrays)

