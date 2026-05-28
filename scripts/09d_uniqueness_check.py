# =============================================================================
#  09d_uniqueness_check.py  —  ¿El conjunto de matchings estables es único
#  en el mercado COMPLETO de Bogotá (10000 x 100)?
#
#  Motivación: la WP-Rule (09c) recupera a DA en mercados pequeños porque el
#  conjunto de matchings estables es casi único. Este script verifica ese
#  hallazgo a ESCALA, sin resolver el ILP de un millón de variables.
#
#  Criterio exacto (Gale-Shapley 1962; Roth-Sotomayor 1990 para many-to-one
#  responsivo): el conjunto de matchings estables es un retículo cuyos extremos
#  son
#     - el matching estable óptimo para ESTUDIANTES = DA con estudiantes
#       proponiendo (matching_utils.deferred_acceptance), y
#     - el matching estable óptimo para COLEGIOS = DA con colegios proponiendo.
#  Si ambos extremos coinciden, el matching estable es ÚNICO.
#
#  Medimos la brecha = número de estudiantes asignados a un colegio distinto
#  entre los dos extremos, sobre varias loterías. Brecha ~ 0 => el retículo
#  está colapsado a un punto, y entonces DA, SED y WP coinciden bajo prioridad
#  fija. Brecha grande => hay margen y la WP podría diferir de DA a escala.
# =============================================================================
import sys, time
from collections import deque
import numpy as np
import pandas as pd

sys.path.insert(0, "/content/edu-matching-paper/scripts")
sys.path.insert(0, ".")
import matching_utils as mu

BASE = "/content/edu-matching-paper/data"


def cargar(world):
    """world in {'bias', 'true'}. Las preferencias ya vienen ordenadas
    (pref_1 = primera opción ... pref_100 = última)."""
    pref = pd.read_parquet(f"{BASE}/primary/sinteticos_b_preferencias_{world}.parquet")
    col  = pd.read_parquet(f"{BASE}/primary/sinteticos_b_colegios.parquet")
    pref_lists = pref.values.tolist()
    school_cap = dict(zip(col["id_establecimiento"],
                          col["capacidad_sintetica"].astype(int)))
    return pref_lists, school_cap, list(col["id_establecimiento"])


def college_proposing_da(pref_lists, school_cap, lottery_rank, stu_rank):
    """DA con COLEGIOS proponiendo -> matching estable óptimo para colegios.

    Espejo de la DA estudiante-proponente: aquí cada colegio propone bajando
    su orden de prioridad, y cada estudiante retiene la única mejor oferta
    recibida (capacidad 1 por estudiante).

    lottery_rank[sid] : array de largo N; lottery_rank[sid][i] = prioridad del
                        estudiante i en el colegio sid (MENOR = MAYOR prioridad).
    stu_rank[i]       : dict {sid: posición de sid en la lista de i} (menor=mejor).
    """
    N = len(pref_lists)
    # orden de propuestas de cada colegio: estudiantes de mayor a menor prioridad
    school_order = {sid: np.argsort(lottery_rank[sid]) for sid in school_cap}
    ptr        = {sid: 0 for sid in school_cap}     # siguiente estudiante a proponer
    held_school = [None] * N                          # colegio que retiene cada estudiante
    held_count  = {sid: 0 for sid in school_cap}      # cupos ocupados por colegio
    queued = set(school_cap.keys())
    active = deque(school_cap.keys())

    while active:
        sid = active.popleft(); queued.discard(sid)
        order = school_order[sid]
        # el colegio propone hasta llenar su cupo o agotar su lista
        while held_count[sid] < school_cap[sid] and ptr[sid] < N:
            i = int(order[ptr[sid]]); ptr[sid] += 1
            if sid not in stu_rank[i]:           # colegio no aceptable para i
                continue
            cur = held_school[i]
            if cur is None:                       # i estaba libre: acepta provisional
                held_school[i] = sid; held_count[sid] += 1
            elif stu_rank[i][sid] < stu_rank[i][cur]:   # i prefiere sid a lo que tenía
                held_school[i] = sid; held_count[sid] += 1
                held_count[cur] -= 1               # cur libera un cupo
                if cur not in queued:              # y vuelve a proponer
                    queued.add(cur); active.append(cur)
            # else: i rechaza la oferta de sid; el while sigue proponiendo
    return held_school


def brecha(a, b):
    return int(sum(1 for x, y in zip(a, b) if x != y))


if __name__ == "__main__":
    t0 = time.time()
    K = 10   # número de loterías

    for world in ["bias", "true"]:
        pref_lists, school_cap, schools = cargar(world)
        N = len(pref_lists)
        cap_total = sum(school_cap.values())
        stu_rank = [{sid: r for r, sid in enumerate(pl)} for pl in pref_lists]  # una vez

        print(f"\n=== mundo {world}: N={N}, colegios={len(schools)}, "
              f"cupos totales={cap_total} "
              f"({'holgura' if cap_total >= N else 'ESCASEZ'}) ===")

        gaps, sin_asignar = [], []
        for k in range(K):
            rng = np.random.default_rng(1000 + k)
            # lotería por colegio (MTB): prioridad estricta y aleatoria de cada
            # estudiante en cada colegio. Estricta => aplica el criterio del retículo.
            lottery_rank = {sid: rng.permutation(N) for sid in schools}
            prio_fn = lambda i, sid, L=lottery_rank: float(L[sid][i])

            mu_S = mu.deferred_acceptance(pref_lists, school_cap, prio_fn)        # estudiante-óptimo
            mu_C = college_proposing_da(pref_lists, school_cap, lottery_rank, stu_rank)  # colegio-óptimo

            # red de seguridad: ambos extremos DEBEN ser estables
            bp_S = mu.count_blocking_pairs(mu_S, pref_lists, school_cap, prio_fn)
            bp_C = mu.count_blocking_pairs(mu_C, pref_lists, school_cap, prio_fn)
            assert bp_S == 0, f"[loteria {k}] DA estudiante NO estable: {bp_S} blocking pairs"
            assert bp_C == 0, f"[loteria {k}] DA colegio NO estable: {bp_C} blocking pairs"

            gaps.append(brecha(mu_S, mu_C))
            sin_asignar.append(sum(x is None for x in mu_S))

        gaps = np.array(gaps)
        print(f"  sin asignar (DA estudiante): {np.mean(sin_asignar):.0f} de {N}")
        print(f"  brecha estudiante-óptimo vs colegio-óptimo "
              f"(# estudiantes con colegio distinto):")
        print(f"     media={gaps.mean():.1f}  mediana={np.median(gaps):.0f}  "
              f"min={gaps.min()}  max={gaps.max()}  "
              f"({100 * gaps.mean() / N:.3f}% de N)")
        print(f"  loterías con matching estable ÚNICO (brecha=0): "
              f"{100 * (gaps == 0).mean():.0f}%")

    print(f"\n[tiempo: {time.time() - t0:.1f}s]")
    print("Lectura: brecha cercana a 0 => el retículo de matchings estables está")
    print("colapsado a un punto, por lo que DA, SED y WP coinciden bajo prioridad")
    print("fija. Confirma a escala el hallazgo de los mercados pequeños. Brecha")
    print("grande => hay múltiples matchings estables y la WP podría diferir de DA.")
