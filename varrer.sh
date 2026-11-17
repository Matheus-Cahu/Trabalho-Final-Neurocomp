#!/usr/bin/env bash
# =====================================================================
#  varrer.sh — roda as varreduras do neuro_redes e gera as figuras
#
#  Uso:
#     ./varrer.sh fig3                     # neurônio isolado (eps = 0)
#     ./varrer.sh fig4 barabasi 10         # 9 rasters, estimulando os 10 maiores hubs
#     ./varrer.sh fig5 bianconi 10         # mapas R, CV, F
#     ./varrer.sh fig5 random              # na rede random todos são estimulados
#     ./varrer.sh fig5 barabasi 10 --teste # só mostra quantas simulações faltam
#
#  Em segundo plano (continua se fechar o terminal):
#     nohup ./varrer.sh fig5 barabasi 10 > log_fig5_barabasi.txt 2>&1 &
#
#  Pontos que já estão no CSV são pulados: se a varredura parar no meio,
#  rode o mesmo comando de novo e ela continua de onde parou.
# =====================================================================

set -euo pipefail
export LC_ALL=C                      # ponto decimal em seq/awk/printf

# ------------------------- configuração ------------------------------
EXE=./neuro_redes
CSV=resultados_redes.csv
N=100                                # deve ser o N com que o executável foi compilado
NUCLEOS=${NUCLEOS:-$(nproc)}                     # use $(( $(nproc) - 1 )) para deixar um núcleo livre

# Fig. 3 — neurônio isolado: g_ext x nu, eps = 0
FIG3_GEXT=$(seq 0 0.1 1)
FIG3_NU=$(seq 0.125 0.125 2)
FIG3_TLIM=2000                       # 1 s de transiente + 1 s de contagem (como no artigo)

# Fig. 4 — rasters
FIG4_GEXT="0.1 0.5 1.0"
FIG4_EPS="0.01 0.1 1"

# Fig. 5 — rede: eps (log) x g_ext
FIG5_GEXT=$(seq 0 0.1 1)                                             # 11 valores
FIG5_EPS=$(awk 'BEGIN{n=10; for(k=0;k<n;k++) printf "%.6g ", 10^(-3+3*k/(n-1))}')  # 10 valores
FIG5_SEEDS=$(seq 1 3)                                                # realizações

TLIM=${TLIM:-11000}                           # duração das simulações da rede (artigo: 11 s)
NU=1.0
# ---------------------------------------------------------------------

TESTE=0
POS=()
for arg in "$@"; do
    if [[ "$arg" == "--teste" ]]; then TESTE=1; else POS+=("$arg"); fi
done
FIG=${POS[0]:-}
REDE=${POS[1]:-random}
HUBS=${POS[2]:-}
if [[ "$REDE" == random ]]; then
    HUBS=$N                          # na rede random todos são estimulados
elif [[ -z "$HUBS" ]]; then
    echo "informe o número de hubs estimulados: $0 $FIG $REDE <n_hubs>"; exit 1
fi

case "$REDE" in random|barabasi|bianconi) ;; *)
    echo "rede inválida: $REDE (use random, barabasi ou bianconi)"; exit 1;; esac
[[ -x "$EXE" ]] || { echo "executável $EXE não encontrado; compile antes:"; \
    echo "  gcc -O2 neuro_redes.c -o neuro_redes -lm"; exit 1; }

# Gera a lista de simulações: "gExt eps seed nu tlim salvar rede", uma por linha
lista() {
    case "$FIG" in
    fig3)
        for g in $FIG3_GEXT; do for nu in $FIG3_NU; do
            echo "$g 0 1 $nu $FIG3_TLIM 0 $REDE $HUBS"
        done; done ;;
    fig4)
        for g in $FIG4_GEXT; do for e in $FIG4_EPS; do
            echo "$g $e 1 $NU $TLIM 1 $REDE $HUBS"
        done; done ;;
    fig5)
        for g in $FIG5_GEXT; do for e in $FIG5_EPS; do for s in $FIG5_SEEDS; do
            echo "$g $e $s $NU $TLIM 0 $REDE $HUBS"
        done; done; done ;;
    *)
        echo "uso: $0 fig3|fig4|fig5 [random|barabasi|bianconi] [--teste]" >&2; exit 1 ;;
    esac
}

# Remove da lista o que já está no CSV (mesmo formato que o neuro_redes grava)
pendentes() {
    while read -r g e s nu t salvar rede h; do
        chave=$(printf "%s,%d,%d,%.4f,%.6f,%.3f,%d,%.0f," "$rede" "$N" "$h" "$g" "$e" "$nu" "$s" "$t")
        if [[ -f "$CSV" ]] && grep -q "^$chave" "$CSV"; then
            # na fig4 também é preciso que o raster exista
            [[ "$salvar" == 1 ]] && ! ls rasters/raster_${rede}_N${N}_h${h}_g$(printf "%.3f" "$g")_e$(printf "%.4f" "$e")_*_s${s}.txt >/dev/null 2>&1 \
                && echo "$g $e $s $nu $t $salvar $rede $h"
            continue
        fi
        echo "$g $e $s $nu $t $salvar $rede $h"
    done
}

TOTAL=$(lista | wc -l)
FALTAM=$(lista | pendentes | wc -l)
echo "$(date '+%F %T')  $FIG, rede $REDE, $HUBS estimulados: $TOTAL simulações no total, $FALTAM a rodar, em $NUCLEOS núcleos"

if [[ $TESTE == 1 ]]; then exit 0; fi

if [[ $FALTAM -gt 0 ]]; then
    lista | pendentes | xargs -P "$NUCLEOS" -L 1 sh -c \
        "$EXE \$0 \$1 \$2 \$3 \$4 \$5 \$6 \$7 > /dev/null 2>> saida_${FIG}_${REDE}_h${HUBS}.log"
fi
echo "$(date '+%F %T')  simulações concluídas"

# Figura
case "$FIG" in
    fig3) OPC="--csv $CSV --N $N --rede $REDE --hubs $HUBS --tlim $FIG3_TLIM" ;;
    fig4) OPC="--rede $REDE --N $N --hubs $HUBS" ;;
    fig5) OPC="--csv $CSV --N $N --rede $REDE --hubs $HUBS --tlim $TLIM" ;;
esac
python3 plot.py "$FIG" $OPC

command -v notify-send >/dev/null && notify-send "Simulação" "$FIG ($REDE, $HUBS estimulados) pronta" || true
