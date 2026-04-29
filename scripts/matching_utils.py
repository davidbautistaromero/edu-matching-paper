"""
matching_utils.py
=================
Módulo compartido con los algoritmos de matching escolar y las métricas de evaluación.
Importado por 07_matching_bm_da.py (datos reales) y 09_matching_sinteticos.py (datos sintéticos).

Algoritmos implementados
------------------------
boston_mechanism     — Boston Mechanism (BM): propuestas irrevocables por ronda.
deferred_acceptance  — Deferred Acceptance / Gale-Shapley (DA): propuestas provisionales,
                       strategyproof del lado del estudiante (Roth 1982).

Abstracción de prioridad
------------------------
Ambos algoritmos reciben una función `priority_fn(student_idx, school_id) -> float`
donde un valor MENOR indica MAYOR prioridad del estudiante sobre el colegio.

    Datos reales (07)  : priority_fn = distancia Haversine en km
                         → colegio prefiere al estudiante más cercano
    Datos sintéticos (09): priority_fn = rango de lotería aleatoria
                         → prioridad uniforme sin información geográfica

Esta abstracción permite reutilizar exactamente el mismo código de matching
en ambos contextos cambiando sólo la función de prioridad.

Métricas implementadas
----------------------
compute_metrics   — eficiencia, equidad, sesgo visual, blocking pairs, rank obtenido.
                    Acepta columnas configurables (quality_col, visual_col) para
                    adaptarse a datos reales (q_j, sobre_demanda_j) o sintéticos
                    (q_j_std, v_j).

Referencias
-----------
- Gale, D. & Shapley, L. (1962). College Admissions and the Stability of Marriage.
  American Mathematical Monthly, 69(1), 9–15.
- Roth, A. (1982). The Economics of Matching: Stability and Incentives.
  Mathematics of Operations Research, 7(4), 617–628.
- Abdulkadiroğlu, A. & Sönmez, T. (2003). School Choice: A Mechanism Design Approach.
  American Economic Review, 93(3), 729–747.
"""

import heapq
import logging
from collections import defaultdict
from typing import Callable

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Boston Mechanism
# ─────────────────────────────────────────────────────────────────────────────

def boston_mechanism(
    pref_lists: list[list[str]],
    school_cap: dict[str, int],
    priority_fn: Callable[[int, str], float],
) -> list[str | None]:
    """
    Implementa el Boston Mechanism (BM) para asignación escolar.

    Descripción del algoritmo
    -------------------------
    El BM opera en rondas sucesivas. En cada ronda k:
      1. Cada estudiante no asignado propone al k-ésimo colegio de su lista
         (saltando colegios sin cupos disponibles).
      2. Cada colegio recibe todas las propuestas de la ronda y acepta de forma
         DEFINITIVA a los top-cap estudiantes según priority_fn.
         Los rechazados pasan a la ronda k+1 con su siguiente preferencia.
      3. El proceso termina cuando no quedan propuestas pendientes (todos asignados,
         sin cupos en ningún colegio preferido, o listas agotadas).

    Propiedad clave: la aceptación es IRREVOCABLE. Un colegio que acepta a un
    estudiante en ronda k no puede rechazarlo en rondas posteriores aunque llegue
    un estudiante con mayor prioridad. Esto hace que el BM NO sea strategyproof:
    un estudiante puede beneficiarse ocultando su primera preferencia.

    Complejidad: O(N × K × log(cap_max)) donde K = longitud máxima de lista.

    Parameters
    ----------
    pref_lists : list[list[str]]
        Lista de N listas de preferencias ordenadas (mejor → peor).
        pref_lists[i] = preferencias del estudiante i.
    school_cap : dict[str, int]
        Capacidad de cada colegio. Llave = id_establecimiento.
    priority_fn : callable(int, str) -> float
        Función de prioridad: priority_fn(student_idx, school_id) devuelve un
        número donde MENOR = MAYOR prioridad del estudiante sobre el colegio.

    Returns
    -------
    assignment : list[str | None]
        assignment[i] = id_establecimiento asignado al estudiante i,
        o None si agotó su lista sin ser asignado.
    """
    N = len(pref_lists)

    # Capacidades restantes: se decrementan con cada aceptación definitiva
    remaining_cap = dict(school_cap)

    # Resultado final: assignment[i] = school_id asignado (None = no asignado)
    assignment = [None] * N

    # Conjunto de estudiantes aún sin asignar (índices 0..N-1)
    unmatched_set = set(range(N))

    # Puntero a la siguiente preferencia de cada estudiante.
    # pref_ptr[i] indica el índice en pref_lists[i] del próximo colegio a proponer.
    pref_ptr = [0] * N

    while unmatched_set:
        # ── Fase 1: cada estudiante no asignado elige su próxima propuesta ──────
        proposals = {}   # school_id → [student_idx, ...]  propuestas de esta ronda
        exhausted = []   # estudiantes que agotaron su lista

        for i in list(unmatched_set):
            # Avanzar pref_ptr[i] hasta encontrar un colegio con cupos disponibles
            while pref_ptr[i] < len(pref_lists[i]):
                sid = pref_lists[i][pref_ptr[i]]
                if remaining_cap.get(sid, 0) > 0:
                    # Este colegio tiene cupo: proponer aquí esta ronda
                    proposals.setdefault(sid, []).append(i)
                    break
                # Sin cupo → no tiene sentido proponer; avanzar al siguiente
                pref_ptr[i] += 1
            else:
                # El bucle terminó sin break: se agotaron todas las preferencias
                exhausted.append(i)

        # Estudiantes sin opciones restantes: sacarlos del pool definitivamente
        for i in exhausted:
            unmatched_set.discard(i)

        # Si no hubo ninguna propuesta válida en esta ronda, el proceso termina
        if not proposals:
            break

        # ── Fase 2: cada colegio decide a quién aceptar (irrevocablemente) ──────
        for sid, applicants in proposals.items():
            cap = remaining_cap[sid]

            if len(applicants) <= cap:
                # Todos los postulantes caben: aceptar a todos
                accepted = applicants
            else:
                # Más postulantes que cupos: ordenar por prioridad (ascendente)
                # y tomar los primeros cap estudiantes
                applicants.sort(key=lambda i: priority_fn(i, sid))
                accepted = applicants[:cap]

            for i in accepted:
                assignment[i] = sid          # aceptación definitiva
                unmatched_set.discard(i)
                remaining_cap[sid] -= 1      # descontar un cupo del colegio

            # Los rechazados deben proponer a su siguiente preferencia en la próxima ronda
            rejected = set(applicants) - set(accepted)
            for i in rejected:
                pref_ptr[i] += 1   # avanzar el puntero para no proponer aquí de nuevo

    return assignment


# ─────────────────────────────────────────────────────────────────────────────
# Deferred Acceptance (Gale-Shapley, estudiante-proponente)
# ─────────────────────────────────────────────────────────────────────────────

def deferred_acceptance(
    pref_lists: list[list[str]],
    school_cap: dict[str, int],
    priority_fn: Callable[[int, str], float],
) -> list[str | None]:
    """
    Implementa Deferred Acceptance / Gale-Shapley (DA) estudiante-proponente.

    Descripción del algoritmo
    -------------------------
    DA opera en rondas hasta que no haya más rechazos:
      1. Cada estudiante no asignado propone al siguiente colegio de su lista.
      2. Cada colegio recibe las propuestas y forma un holding provisional de
         hasta cap estudiantes según priority_fn. Los estudiantes por encima
         del límite (los de menor prioridad) son RECHAZADOS y vuelven al pool.
      3. El rechazo puede ser de un estudiante previamente aceptado si llega
         uno con mayor prioridad (Deferred = la aceptación es provisional).
      4. El proceso termina cuando ningún estudiante es rechazado en una ronda
         (el matching provisional se vuelve definitivo).

    Propiedades clave:
      - STRATEGYPROOF (lado estudiante): reportar preferencias verdaderas es
        estrategia dominante — ningún estudiante puede mejorar mintiendo
        (Roth 1982). Por eso DA es resistente al sesgo estratégico.
      - ESTABLE: el matching resultante no tiene blocking pairs.
        Si (i, j) fuera un blocking pair, significaría que j prefiere a i sobre
        alguno de sus asignados actuales, pero DA garantiza que j habría aceptado
        a i si hubiera propuesto (y lo habría expulsado a quien desplaza).
      - STUDENT-OPTIMAL: entre todos los matchings estables, DA produce el
        mejor para los estudiantes (Gale & Shapley 1962).

    Implementación con max-heap por colegio:
      Cada colegio mantiene un heap de sus holding provisionales, ordenado por
      prioridad descendente (el peor estudiante al tope del heap). Cuando llega
      un nuevo postulante, se compara con el peor: si tiene mejor prioridad,
      se hace el intercambio en O(log cap). Esto da O(N·K·log cap) total.

    Parameters
    ----------
    pref_lists : list[list[str]]
        Lista de N listas de preferencias ordenadas.
    school_cap : dict[str, int]
        Capacidad de cada colegio.
    priority_fn : callable(int, str) -> float
        Función de prioridad: MENOR valor = MAYOR prioridad.

    Returns
    -------
    assignment : list[str | None]
        assignment[i] = school_id asignado (None si agotó lista sin match).
    """
    N = len(pref_lists)

    # Puntero de propuesta por estudiante (índice en pref_lists[i])
    pref_ptr = [0] * N

    # Resultado final; se construye al final desde los holdings
    assignment = [None] * N

    # holding[sid]: min-heap de (-priority_val, student_idx).
    # Usamos -priority_val porque heapq es un min-heap, pero queremos que
    # el PEOR estudiante (mayor priority_val) quede en el tope → negamos el valor.
    holding = {sid: [] for sid in school_cap}

    # holding_set[sid]: conjunto de índices en el holding de sid (para lookup O(1))
    holding_set = {sid: set() for sid in school_cap}

    # Pool inicial: todos los estudiantes sin asignar
    unmatched = set(range(N))

    while unmatched:
        new_unmatched = set()   # estudiantes expulsados en esta ronda
        progress = False         # flag para detectar si hubo al menos una propuesta

        for i in list(unmatched):
            # Si agotó su lista, ya no puede proponer a ningún colegio nuevo
            if pref_ptr[i] >= len(pref_lists[i]):
                continue

            # Proponer al siguiente colegio en la lista
            sid = pref_lists[i][pref_ptr[i]]
            pref_ptr[i] += 1   # avanzar para no proponer aquí de nuevo
            progress = True

            pri_i = priority_fn(i, sid)   # valor de prioridad (menor = mejor)
            heap  = holding[sid]
            h_set = holding_set[sid]

            if len(heap) < school_cap[sid]:
                # El colegio tiene cupo libre: aceptar provisionalmente
                # Guardamos -pri_i para que el peor (mayor pri) quede al tope
                heapq.heappush(heap, (-pri_i, i))
                h_set.add(i)

            else:
                # El colegio está lleno: comparar con el peor provisional actual
                worst_neg_pri, worst_idx = heap[0]   # tope = peor estudiante
                worst_pri = -worst_neg_pri            # revertir la negación

                if pri_i < worst_pri:
                    # El nuevo tiene MEJOR prioridad que el peor provisional:
                    # expulsar al peor y aceptar al nuevo
                    heapq.heapreplace(heap, (-pri_i, i))   # reemplaza el tope
                    h_set.discard(worst_idx)
                    h_set.add(i)
                    # El expulsado vuelve al pool; su pref_ptr NO se resetea,
                    # así continuará desde el siguiente colegio en su lista
                    new_unmatched.add(worst_idx)
                # else: el nuevo tiene PEOR prioridad → rechazado directamente
                # Su pref_ptr ya fue avanzado; propondrá al siguiente en la
                # próxima iteración si sigue sin asignar

        # Si no hubo ninguna propuesta válida, el proceso ha convergido
        if not progress:
            break

        # Reconstruir el pool de no asignados para la siguiente ronda:
        # son los estudiantes que (a) no están en ningún holding, Y
        # (b) aún tienen colegios a los que proponer
        assigned_now = set().union(*holding_set.values())
        unmatched = (
            {i for i in range(N)
             if i not in assigned_now and pref_ptr[i] < len(pref_lists[i])}
            | (new_unmatched - assigned_now)   # expulsados que aún tienen opciones
        )

    # Convertir holdings a assignment final
    for sid, h_set in holding_set.items():
        for i in h_set:
            assignment[i] = sid

    return assignment


# ─────────────────────────────────────────────────────────────────────────────
# Métricas de evaluación
# ─────────────────────────────────────────────────────────────────────────────

def count_blocking_pairs(
    assignment: list[str | None],
    pref_lists: list[list[str]],
    school_cap: dict[str, int],
    priority_fn: Callable[[int, str], float],
) -> int:
    """
    Cuenta el número de blocking pairs en un matching dado.

    Definición de blocking pair (i, j):
      1. Estudiante i prefiere el colegio j a su asignación actual (o está sin asignar).
      2. El colegio j prefiere a i sobre alguno de sus estudiantes actuales,
         O el colegio j tiene cupos libres (en cuyo caso j "prefiere" a i
         sobre nadie — cualquier asignación adicional lo deja igual o mejor).

    Un matching sin blocking pairs es ESTABLE. DA siempre produce 0 blocking
    pairs por construcción; BM puede tener muchos porque sus aceptaciones
    irrevocables en rondas tempranas pueden crear bloqueos en rondas posteriores.

    Algoritmo:
      Para cada estudiante i, identificar los colegios que prefiere a su asignado.
      Para cada uno de esos colegios j, verificar si j tiene cupo libre o si
      i tiene mayor prioridad que el peor estudiante actualmente en j.

    Complejidad: O(N × K) donde K = longitud media de lista de preferencias.

    Returns
    -------
    int : número de blocking pairs en el matching.
    """
    # Construir quiénes están asignados a cada colegio (índice → lista de estudiantes)
    holding_idx: dict[str, list[int]] = defaultdict(list)
    for i, sid in enumerate(assignment):
        if sid is not None:
            holding_idx[sid].append(i)

    # Para cada colegio, calcular la prioridad del peor estudiante asignado.
    # "Peor" = mayor valor de priority_fn. Este es el candidato a desplazar.
    worst_pri: dict[str, float] = {
        sid: max(priority_fn(i, sid) for i in students)
        for sid, students in holding_idx.items()
        if students
    }
    current_n: dict[str, int] = {sid: len(sts) for sid, sts in holding_idx.items()}

    bp = 0
    for i, prefs_i in enumerate(pref_lists):
        assigned_sid = assignment[i]

        # Rango del colegio asignado en la lista de preferencias de i.
        # Un estudiante sin asignar "prefiere" cualquier colegio (rank = len(lista))
        if assigned_sid is None:
            rank_assigned = len(prefs_i)
        else:
            try:
                rank_assigned = prefs_i.index(assigned_sid)
            except ValueError:
                # El colegio asignado no está en la lista (caso raro en datos reales
                # cuando la lista fue truncada): asumir posición al final
                rank_assigned = len(prefs_i)

        # Examinar sólo los colegios que i prefiere sobre su asignado
        for sid_j in prefs_i[:rank_assigned]:
            cap_j = school_cap.get(sid_j, 0)
            n_j   = current_n.get(sid_j, 0)

            if n_j < cap_j:
                # Colegio con cupo libre → es un blocking pair (j aceptaría a i)
                bp += 1
                break

            # Colegio lleno: verificar si i desplazaría al peor estudiante
            pri_i_j = priority_fn(i, sid_j)
            if pri_i_j < worst_pri.get(sid_j, np.inf):
                # i tiene mejor prioridad que el peor actual → blocking pair
                bp += 1
                break
            # i no desplaza a nadie en j → no bloquea, revisar siguiente preferencia

    return bp


def mean_rank_obtained(
    assignment: list[str | None],
    pref_lists: list[list[str]],
) -> float:
    """
    Calcula el rango medio (1-based) del colegio asignado en la lista de cada estudiante.

    Un rank_medio cercano a 1 indica que la mayoría obtuvo su primera preferencia.
    Un rank_medio alto indica que muchos estudiantes terminaron en opciones lejanas.
    Los no asignados reciben rank = len(lista) + 1 (penalización).
    """
    ranks = []
    for i, sid in enumerate(assignment):
        if sid is None:
            ranks.append(len(pref_lists[i]) + 1)   # penalización por no asignar
        else:
            try:
                ranks.append(pref_lists[i].index(sid) + 1)   # +1 para base-1
            except ValueError:
                ranks.append(len(pref_lists[i]) + 1)
    return float(np.mean(ranks))


def compute_metrics(
    assignment: list[str | None],
    pref_lists: list[list[str]],
    school_cap: dict[str, int],
    priority_fn: Callable[[int, str], float],
    school_info: pd.DataFrame,
    quality_col: str,
    visual_col: str,
    estrato_arr: np.ndarray,
    label: str = "",
    localidad_arr: np.ndarray | None = None,
) -> dict:
    """
    Calcula el conjunto completo de métricas de evaluación de un matching.

    Framework de métricas
    ---------------------
    1. EFICIENCIA (sentido Pareto)
       rank_medio    : posición media (1-based) del colegio asignado en la lista
                       declarada de cada familia. rank=1 → obtuvo su primera opción.
                       Fundamentado en Abdulkadiroğlu & Sönmez (2003): la comparación
                       de mecanismos se hace sobre el rank obtenido respecto a prefs
                       verdaderas. No depende de ninguna variable externa.

    2. EQUIDAD
       equidad_aj    : correlación de Pearson entre estrato y a_j del colegio asignado.
                       a_j captura el atractivo compuesto revelado por la demanda
                       (calidad académica + entorno visual + seguridad + transporte).
                       Correlación ≈ 0 → el mecanismo distribuye el atractivo sin
                       sesgo socioeconómico.
       aj_estrato_{s}: a_j media asignada a familias de estrato s. Permite ver la
                       distribución desagregada del atractivo.

    3. SESGO VISUAL
       sesgo_visual  : correlación de Pearson entre estrato y visual_col (v_j) del
                       colegio asignado. Mide si el atractivo visual específico —
                       la contribución metodológica del paper — favorece estratos altos.

    4. CALIDAD ACADÉMICA (política pública)
       qj_medio      : media de quality_col (q_j) entre familias asignadas.
                       Variable de bienestar de política: ¿los estudiantes acceden
                       a colegios académicamente buenos?
       qj_estrato_{s}: q_j media por estrato. ¿Los estratos bajos acceden a calidad?

    5. DÉFICIT SECTORIZADO
       rechazo_total    : fracción de familias sin asignación en la población total.
       rechazo_estrato_{s}: fracción de familias sin asignación por estrato s.
       Si localidad_arr se provee: rechazo_loc_{nombre} por localidad.

    Parameters
    ----------
    assignment : list[str | None]
        Resultado de boston_mechanism o deferred_acceptance.
    pref_lists : list[list[str]]
        Listas de preferencias (para rank_medio y blocking_pairs).
    school_cap : dict[str, int]
        Capacidades (para blocking_pairs).
    priority_fn : callable
        Función de prioridad (para blocking_pairs).
    school_info : pd.DataFrame
        Indexado por school_id. Debe contener quality_col, visual_col y "a_j".
    quality_col : str
        Columna de calidad académica (ej. "q_j", "q_j_std").
    visual_col : str
        Columna de señal visual pura (ej. "v_j").
    estrato_arr : np.ndarray
        Estrato socioeconómico de cada familia (1–6).
    label : str
        Etiqueta del mecanismo (ej. "BM", "DA").
    localidad_arr : np.ndarray | None
        Localidad de cada familia (nombre string). Si se provee, calcula
        tasas de rechazo por localidad.

    Returns
    -------
    dict con todas las métricas descritas.
    """
    N            = len(assignment)
    matched_mask = np.array([a is not None for a in assignment])

    # ── Extraer atributos del colegio asignado ────────────────────────────────
    # a_j: atractivo compuesto (siempre desde school_info["a_j"] si existe)
    has_aj = "a_j" in school_info.columns

    def get_col(col):
        return np.array([
            school_info.loc[a, col]
            if (a is not None and a in school_info.index and col in school_info.columns)
            else np.nan
            for a in assignment
        ])

    q_assigned  = get_col(quality_col)
    v_assigned  = get_col(visual_col)
    aj_assigned = get_col("a_j") if has_aj else q_assigned  # fallback a quality_col

    # ── Subconjunto matched ───────────────────────────────────────────────────
    q_m  = q_assigned[matched_mask]
    v_m  = v_assigned[matched_mask]
    aj_m = aj_assigned[matched_mask]
    s_m  = estrato_arr[matched_mask]

    def safe_corr(x, y):
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < 2:
            return np.nan
        return float(np.corrcoef(x[mask], y[mask])[0, 1])

    # ── 1. Eficiencia (Pareto) ────────────────────────────────────────────────
    rank_med = mean_rank_obtained(assignment, pref_lists)
    bp       = count_blocking_pairs(assignment, pref_lists, school_cap, priority_fn)

    # ── 2. Equidad (a_j) ─────────────────────────────────────────────────────
    equidad_aj = safe_corr(s_m, aj_m)
    aj_by_s    = {}
    for s in range(1, 7):
        mask_s = (estrato_arr == s) & matched_mask
        aj_by_s[s] = float(np.nanmean(aj_assigned[mask_s])) if mask_s.sum() > 0 else np.nan

    # ── 3. Sesgo visual (v_j) ─────────────────────────────────────────────────
    sesgo_visual = safe_corr(s_m, v_m)

    # ── 4. Calidad académica (q_j) ────────────────────────────────────────────
    qj_medio  = float(np.nanmean(q_m))
    qj_by_s   = {}
    for s in range(1, 7):
        mask_s = (estrato_arr == s) & matched_mask
        qj_by_s[s] = float(np.nanmean(q_assigned[mask_s])) if mask_s.sum() > 0 else np.nan

    # ── 5. Déficit sectorizado ────────────────────────────────────────────────
    rechazo_total = float((~matched_mask).sum() / N) if N > 0 else np.nan

    rechazo_by_s = {}
    for s in range(1, 7):
        mask_s = estrato_arr == s
        n_s    = mask_s.sum()
        n_rec  = (~matched_mask[mask_s]).sum() if n_s > 0 else 0
        rechazo_by_s[s] = float(n_rec / n_s) if n_s > 0 else np.nan

    rechazo_by_loc = {}
    if localidad_arr is not None:
        for loc in np.unique(localidad_arr):
            mask_l  = localidad_arr == loc
            n_l     = mask_l.sum()
            n_rec_l = (~matched_mask[mask_l]).sum() if n_l > 0 else 0
            rechazo_by_loc[str(loc)] = float(n_rec_l / n_l) if n_l > 0 else np.nan

    # ── Ensamblar métricas ────────────────────────────────────────────────────
    metrics = {
        "condicion"        : label,
        "n_asignados"      : int(matched_mask.sum()),
        "n_sin_asignar"    : int((~matched_mask).sum()),
        # Eficiencia Pareto
        "rank_medio"       : round(rank_med, 3),
        "blocking_pairs"   : bp,
        # Equidad (a_j compuesto)
        "equidad_aj"       : round(equidad_aj, 4) if not np.isnan(equidad_aj) else np.nan,
        **{f"aj_estrato_{s}": round(aj_by_s[s], 4) if not np.isnan(aj_by_s[s]) else np.nan
           for s in range(1, 7)},
        # Sesgo visual
        "sesgo_visual"     : round(sesgo_visual, 4) if not np.isnan(sesgo_visual) else np.nan,
        # Calidad académica
        "qj_medio"         : round(qj_medio, 4),
        **{f"qj_estrato_{s}": round(qj_by_s[s], 4) if not np.isnan(qj_by_s[s]) else np.nan
           for s in range(1, 7)},
        # Déficit
        "rechazo_total"    : round(rechazo_total, 4),
        **{f"rechazo_estrato_{s}": round(rechazo_by_s[s], 4) if not np.isnan(rechazo_by_s[s]) else np.nan
           for s in range(1, 7)},
        **{f"rechazo_loc_{k}": round(v, 4) for k, v in rechazo_by_loc.items()},
        # Backward-compat (scripts de Jhoan usan estos nombres)
        "eficiencia_q"     : round(qj_medio, 4),
        "equidad_corr"     : round(equidad_aj, 4) if not np.isnan(equidad_aj) else np.nan,
    }

    log.info(
        f"  [{label}] asignados={metrics['n_asignados']:,} | "
        f"rank={rank_med:.2f} | BP={bp:,} | "
        f"equidad_aj={equidad_aj:+.4f} | sesgo_v={sesgo_visual:+.4f} | "
        f"qj={qj_medio:.3f} | rechazo={rechazo_total:.1%}"
    )

    return metrics
