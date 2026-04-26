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
) -> dict:
    """
    Calcula el conjunto completo de métricas de evaluación de un matching.

    Métricas
    --------
    eficiencia_q  : calidad media (quality_col) del colegio asignado a los estudiantes
                    matched. Refleja qué tan bien el mecanismo maximiza calidad agregada.

    equidad_corr  : correlación de Pearson entre estrato del estudiante y calidad
                    quality_col de su colegio asignado. Un sistema equitativo debería
                    tener esta correlación ≈ 0 (acceso a calidad independiente del estrato).

    sesgo_visual  : correlación de Pearson entre estrato y visual_col del colegio
                    asignado. Si estratos altos terminan en colegios más visualmente
                    atractivos (mayor v_j o sobre_demanda), hay sesgo visual en la asignación.

    rank_medio    : posición media (1-based) del colegio asignado en la lista de prefs.
                    Mide cuán bien se respetan las preferencias declaradas.

    blocking_pairs: número de pares (estudiante, colegio) que harían preferir
                    mutuamente cambiar el matching. DA = 0, BM > 0 generalmente.

    q_estrato_{s} : calidad media asignada a estudiantes de estrato s (s=1..6).
                    Permite analizar distribución de calidad por estrato.

    Parameters
    ----------
    assignment : list[str | None]
        Resultado de boston_mechanism o deferred_acceptance.
    pref_lists : list[list[str]]
        Listas de preferencias usadas en el matching (necesarias para rank y BP).
    school_cap : dict[str, int]
        Capacidades (necesario para contar blocking pairs).
    priority_fn : callable
        Función de prioridad (necesaria para blocking pairs).
    school_info : pd.DataFrame
        DataFrame indexado por school_id con al menos quality_col y visual_col.
    quality_col : str
        Columna de calidad (ej. "q_j" para datos reales, "q_j_std" para sintéticos).
    visual_col : str
        Columna de señal visual (ej. "sobre_demanda_j" o "v_j").
    estrato_arr : np.ndarray
        Estrato socioeconómico de cada estudiante (índice = posición en assignment).
    label : str
        Etiqueta para logging (ej. "BM", "DA", "BM-bias").

    Returns
    -------
    dict con todas las métricas descritas.
    """
    N            = len(assignment)
    matched_mask = np.array([a is not None for a in assignment])

    # Extraer calidad y señal visual del colegio asignado a cada estudiante
    # np.nan para los no asignados (excluidos de correlaciones)
    q_assigned = np.array([
        school_info.loc[a, quality_col] if (a is not None and a in school_info.index)
        else np.nan
        for a in assignment
    ])
    v_assigned = np.array([
        school_info.loc[a, visual_col] if (a is not None and a in school_info.index)
        else np.nan
        for a in assignment
    ])

    # Restringir al subconjunto de estudiantes matched para las correlaciones
    q_matched = q_assigned[matched_mask]
    v_matched = v_assigned[matched_mask]
    s_matched = estrato_arr[matched_mask]

    eficiencia = float(np.nanmean(q_matched))

    # Correlaciones: usar np.corrcoef que devuelve la matriz de correlación 2×2;
    # el elemento [0,1] es la correlación entre las dos variables
    corr_eq = float(np.corrcoef(s_matched, q_matched)[0, 1]) if len(s_matched) > 1 else np.nan
    corr_vj = float(np.corrcoef(s_matched, v_matched)[0, 1]) if len(s_matched) > 1 else np.nan

    # Calidad media por estrato (para analizar distribución desagregada)
    q_by_strato = {}
    for s in range(1, 7):
        mask_s = (estrato_arr == s) & matched_mask
        q_by_strato[s] = float(np.nanmean(q_assigned[mask_s])) if mask_s.sum() > 0 else np.nan

    # Rank medio: posición del colegio asignado en la lista del estudiante
    rank_med = mean_rank_obtained(assignment, pref_lists)

    # Blocking pairs: pares (estudiante, colegio) mutuamente preferibles
    bp = count_blocking_pairs(assignment, pref_lists, school_cap, priority_fn)

    metrics = {
        "condicion"      : label,
        "n_asignados"    : int(matched_mask.sum()),
        "n_sin_asignar"  : int((~matched_mask).sum()),
        "eficiencia_q"   : round(eficiencia, 4),
        "equidad_corr"   : round(corr_eq, 4),
        "sesgo_visual"   : round(corr_vj, 4),
        "rank_medio"     : round(rank_med, 3),
        "blocking_pairs" : bp,
        **{f"q_estrato_{s}": round(q_by_strato[s], 4) for s in range(1, 7)},
    }

    log.info(
        f"  [{label}] asignados={metrics['n_asignados']:,} | "
        f"q̄={eficiencia:.3f} | corr_q={corr_eq:+.4f} | "
        f"corr_v={corr_vj:+.4f} | rank={rank_med:.2f} | BP={bp:,}"
    )

    return metrics
