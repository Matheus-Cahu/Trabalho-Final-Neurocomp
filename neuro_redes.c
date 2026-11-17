#define _GNU_SOURCE      // qsort_r (glibc); inclui também drand48

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>   // flock: trava o CSV quando há execuções em paralelo
#include <sys/stat.h>   // mkdir: cria a pasta dos rasters

#ifndef N
#define N 100            // pode ser alterado na compilação: gcc -DN=500 ...
#endif
#define WINDOW 20.0
#define MAX_SPK 24000    // cobre nu = 2 spikes/ms por 11 s
#define K_MED 10         // número médio de conexões por neurônio (fixo para qualquer N)

// Método de Runge-Kutta:
// Aproximar y(t) dado y'(t) = f(t,y), y(t0) = y0
// k1 = f(tn, yn)
// k2 = f(tn + h/2, yn + h/2 * k1)
// k3 = f(tn + h/2, yn + h/2 * k2)
// k4 = f(tn + h, yn + h * k3)
// y_(n+1) = yn + h/6 (k1 + 2k2 + 2k3 + k4)
//
// Modelo de Hodgkin-Huxley:
// Cm dV/dt = -[gNa * m³h(V-ENa)+gkn⁴(V-Ek)+gv(V-Evaz)] + Iext
// dn/dt = an(V)(1-n)-Bn(V)n
// dm/dt = am(V)(1-m)-Bm(V)m
// dh/dt = ah(V)(1-h)-Bh(V)h


typedef struct {
double v, n, m, h, r;
double train[MAX_SPK];
int n_spk;
int k0;
}neuron;

neuron net[N];

// Parâmetros do Modelo
double Cm = 1;
double gNa = 120;
double eNa = 50;
double gK = 36;
double eK = -77;
double gL = 0.3;
double eL = -54.4;
double gExt;
double eps;
int a[N][N];

// Parâmetros da Simulação
double dt = 0.01;
double tlim = 11000;
double t0 = 0;
double nu = 1.0;

// Função de distribuição uniforme
double uniform(void) {
    return drand48();
}

// Função geradora de trems de Poisson
void generate_train(neuron *p, double nu, double T){
  double t = 0.0;
  p->n_spk = 0;
  p->k0 = 0;
  while(p->n_spk<MAX_SPK){
    t += -log(1.0 - drand48())/nu;
    if (t >= T) break;
    p->train[p->n_spk++] = t;
  }
}

// Funções alfa e beta
double an(double v){
  return (0.01 * (v + 55))/(1-exp(-(v+55)/10));
}

double bn(double v){
  return (0.125 * exp(-(v+65)/80));
}

double am(double v){
  return (0.1*(v+40)/(1-exp(-(v+40)/10)));
}

double bm(double v){
  return (4*exp(-(v + 65)/18));
}

double ah(double v){
  return (0.07 * exp(-(v+65)/20));
}

double bh(double v){
  return (1/(1+exp(-(v+35)/10)));
}

// Funções Derivadas
double dn(double V, double n){ return an(V)*(1-n) - bn(V)*n; }
double dm(double V, double m){ return am(V)*(1-m) - bm(V)*m; }
double dh(double V, double h){ return ah(V)*(1-h) - bh(V)*h; }

double dv(double V, double n, double m, double h, double Iext, double Icoup){
    return (-((gNa * (pow(m,3)) * h * (V - eNa)) + (gK * (pow(n,4)) * (V - eK)) + (gL * (V - eL))) + Iext + Icoup) / Cm;
}

double dr(double V, double r) {
    double T = 1.0 / (1.0 + exp(-(V + 20.0)));
    return (1.0/0.4 - 1.0/2.0) * T * (1.0 - r) - r / 2.0;
}

double s(double t){
  if (t < 0) return 0;
    return 0.625 * (exp((-1*t)/2)-exp((-1*t)/0.4));
}

double sum_s(neuron *p, double t) {
    while (p->k0 < p->n_spk && t - p->train[p->k0] > WINDOW)
        p->k0++;

    double soma = 0.0;
    for (int k = p->k0; k < p->n_spk; k++) {
        double age = t - p->train[k];
        if (age < 0) break;
        soma += s(age);
    }
    return soma;
}

int estimulado[N];      // 1 se o neurônio recebe o trem de Poisson externo

double Iext(double V, double t, neuron *p){
  if (!estimulado[p - net]) return 0.0;   // neurônio sem entrada externa
  return gExt * (40 - V) * sum_s(p, t);
}

double sum_ar(int i, const double r[N]) {
    double soma = 0.0;
    for (int j = 0; j < N; j++)
        if (a[i][j]) soma += r[j];
    return soma;
}

double Icoup(int i, double Vi, const double r[N]) {
    return eps * (40 - Vi) * sum_ar(i, r);
}


/* ===================================================================
 *  A PARTIR DAQUI: código novo (RK4, registro de spikes, métricas, main)
 * =================================================================== */

// Parâmetros de medição
double t_trans = 1000.0;   // ms descartados como transiente (artigo: 1 s)
double v_th    = -20.0;    // mV, seção de Poincaré para detectar spikes
double dt_R    = 0.1;      // ms, amostragem de R(t)

// Instantes dos spikes de saída (após o transiente)
#define MAX_OUT 5000
double spk[N][MAX_OUT];
int    n_out[N];
int    overflow = 0;       // vira 1 se algum neurônio estourar MAX_OUT


/* ------------------------- Geração da rede ----------------------- */
//
// Convenção: a[i][j] = 1 se o neurônio j envia sinapse para o neurônio i.
//
// "random"   : Erdős–Rényi DIRIGIDA, como no artigo. Cada a[i][j] (i != j)
//              é sorteado com p = K_MED/(N-1), de modo que o número médio de
//              entradas (e de saídas) por neurônio é K_MED para qualquer N.
//
// "barabasi" : Barabási–Albert. Começa com um grafo completo de m0 = m+1 nós;
//              cada novo nó faz m = K_MED/2 ligações com nós existentes,
//              escolhidos com probabilidade proporcional ao grau k_j.
//
// "bianconi" : Bianconi–Barabási (modelo de fitness). Igual ao BA, mas a
//              probabilidade é proporcional a eta_j * k_j, com fitness
//              eta_j ~ Uniforme(0,1) sorteado uma vez por nó.
//
// BA e BB geram grafos NÃO dirigidos: cada ligação vira sinapse nos dois
// sentidos (a[i][j] = a[j][i] = 1). Assim, entradas = saídas = grau, com
// média ≈ 2m = K_MED. Ao final, os rótulos dos nós são embaralhados para
// que o índice do neurônio não indique a sua idade (sem isso, os hubs
// seriam sempre os de índice mais baixo).

char tipo_rede[16] = "random";
double fitness[N];                 // usado só no modelo de Bianconi–Barabási

int net_type_valid(const char *t) {
    return !strcmp(t, "random") || !strcmp(t, "barabasi") || !strcmp(t, "bianconi");
}

static void build_random(void) {
    double p_con = (double)K_MED / (N - 1);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            a[i][j] = (i != j) && (uniform() < p_con);
}

// Crescimento preferencial (BA quando use_fitness = 0, BB quando = 1)
static void build_preferential(int use_fitness) {
    static int g[N][N];            // grafo não dirigido antes do embaralhamento
    int grau[N];
    int m  = K_MED / 2;
    int m0 = m + 1;

    for (int i = 0; i < N; i++) {
        grau[i] = 0;
        fitness[i] = use_fitness ? uniform() : 1.0;
        for (int j = 0; j < N; j++) g[i][j] = 0;
    }

    // núcleo inicial: grafo completo com m0 nós (cada um com grau m)
    for (int i = 0; i < m0; i++)
        for (int j = i + 1; j < m0; j++) {
            g[i][j] = g[j][i] = 1;
            grau[i]++; grau[j]++;
        }

    // crescimento: o nó novo v escolhe m alvos distintos entre 0..v-1
    for (int v = m0; v < N; v++) {
        int escolhidos = 0;
        while (escolhidos < m) {
            double total = 0.0;
            for (int j = 0; j < v; j++)
                if (!g[v][j]) total += fitness[j] * grau[j];
            if (total <= 0.0) break;   // só ocorre se todos os candidatos tiverem peso 0

            double x = uniform() * total, acc = 0.0;
            int alvo = -1;
            for (int j = 0; j < v; j++) {
                if (g[v][j]) continue;
                acc += fitness[j] * grau[j];
                if (x < acc) { alvo = j; break; }
            }
            if (alvo < 0) continue;    // arredondamento numérico: sorteia de novo

            g[v][alvo] = g[alvo][v] = 1;
            grau[v]++; grau[alvo]++;
            escolhidos++;
        }
    }

    // embaralha os rótulos (Fisher–Yates) e copia para a matriz da rede
    int perm[N];
    for (int i = 0; i < N; i++) perm[i] = i;
    for (int i = N - 1; i > 0; i--) {
        int k = (int)(uniform() * (i + 1));
        int tmp = perm[i]; perm[i] = perm[k]; perm[k] = tmp;
    }
    double fit_perm[N];
    for (int i = 0; i < N; i++) fit_perm[perm[i]] = fitness[i];
    for (int i = 0; i < N; i++) fitness[i] = fit_perm[i];

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            a[perm[i]][perm[j]] = g[i][j];
}

void build_network(const char *tipo) {
    if (!strcmp(tipo, "random"))        build_random();
    else if (!strcmp(tipo, "barabasi")) build_preferential(0);
    else                                build_preferential(1);
}

// Grau de entrada (soma da linha i) e de saída (soma da coluna i)
void degrees(int kin[N], int kout[N]) {
    for (int i = 0; i < N; i++) kin[i] = kout[i] = 0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (a[i][j]) { kin[i]++; kout[j]++; }
}

// Define quem recebe a entrada externa.
//   random            : todos os neurônios (como no artigo).
//   barabasi/bianconi : só os n_hubs neurônios de MAIOR grau. Os neurônios
//                       são ordenados por grau decrescente; empates são
//                       resolvidos pelo índice, que já é aleatório porque os
//                       rótulos foram embaralhados em build_preferential().
int n_hubs = -1;        // -1 = não informado

static int cmp_grau_desc(const void *x, const void *y, void *grau) {
    int i = *(const int *)x, j = *(const int *)y;
    const int *k = (const int *)grau;
    if (k[i] != k[j]) return k[j] - k[i];   // grau maior primeiro
    return i - j;
}

void select_stimulated(void) {
    if (!strcmp(tipo_rede, "random")) {
        for (int i = 0; i < N; i++) estimulado[i] = 1;
        return;
    }
    int kin[N], kout[N], ordem[N];
    degrees(kin, kout);                       // BA/BB: grau de entrada = grau de saída
    for (int i = 0; i < N; i++) { ordem[i] = i; estimulado[i] = 0; }
    qsort_r(ordem, N, sizeof(int), cmp_grau_desc, kin);
    for (int r = 0; r < n_hubs; r++) estimulado[ordem[r]] = 1;

    fprintf(stderr, "estimulados: %d hubs, graus de %d a %d (grau do 1o nao estimulado: %d)\n",
            n_hubs,
            n_hubs > 0 ? kin[ordem[0]] : 0,
            n_hubs > 0 ? kin[ordem[n_hubs-1]] : 0,
            n_hubs < N ? kin[ordem[n_hubs]] : 0);
}

// Resumo da rede no terminal (stderr), para conferência
void network_summary(void) {
    int kin[N], kout[N];
    degrees(kin, kout);
    int kmin = kin[0], kmax = kin[0];
    double soma = 0;
    for (int i = 0; i < N; i++) {
        soma += kin[i];
        if (kin[i] < kmin) kmin = kin[i];
        if (kin[i] > kmax) kmax = kin[i];
    }
    fprintf(stderr, "rede %s: N=%d, grau de entrada medio=%.2f, min=%d, max=%d\n",
            tipo_rede, N, soma / N, kmin, kmax);
}


/* ----------------------------- RK4 ------------------------------- */

// Estado dinâmico de um neurônio (as 5 variáveis integradas)
typedef struct { double v, n, m, h, r; } state;

// Derivadas do neurônio i no estado y, com os r da rede em r_net, no tempo t
static state deriv(int i, state y, const double r_net[N], double t) {
    state d;
    double Ie = Iext(y.v, t, &net[i]);
    double Ic = Icoup(i, y.v, r_net);
    d.v = dv(y.v, y.n, y.m, y.h, Ie, Ic);
    d.n = dn(y.v, y.n);
    d.m = dm(y.v, y.m);
    d.h = dh(y.v, y.h);
    d.r = dr(y.v, y.r);
    return d;
}

// y + c·k
static state add_scaled(state y, state k, double c) {
    state o = { y.v + c*k.v, y.n + c*k.n, y.m + c*k.m, y.h + c*k.h, y.r + c*k.r };
    return o;
}

// Avança a rede inteira de t para t + dt
void rk4_step(double t) {
    static state  y0[N], yt[N], k1[N], k2[N], k3[N], k4[N];
    static double r_net[N];   // r de todos os neurônios no estágio atual

    for (int i = 0; i < N; i++) {
        y0[i] = (state){ net[i].v, net[i].n, net[i].m, net[i].h, net[i].r };
        r_net[i] = y0[i].r;
    }

    for (int i = 0; i < N; i++) k1[i] = deriv(i, y0[i], r_net, t);

    for (int i = 0; i < N; i++) { yt[i] = add_scaled(y0[i], k1[i], dt/2); r_net[i] = yt[i].r; }
    for (int i = 0; i < N; i++) k2[i] = deriv(i, yt[i], r_net, t + dt/2);

    for (int i = 0; i < N; i++) { yt[i] = add_scaled(y0[i], k2[i], dt/2); r_net[i] = yt[i].r; }
    for (int i = 0; i < N; i++) k3[i] = deriv(i, yt[i], r_net, t + dt/2);

    for (int i = 0; i < N; i++) { yt[i] = add_scaled(y0[i], k3[i], dt);   r_net[i] = yt[i].r; }
    for (int i = 0; i < N; i++) k4[i] = deriv(i, yt[i], r_net, t + dt);

    for (int i = 0; i < N; i++) {
        net[i].v = y0[i].v + dt/6*(k1[i].v + 2*k2[i].v + 2*k3[i].v + k4[i].v);
        net[i].n = y0[i].n + dt/6*(k1[i].n + 2*k2[i].n + 2*k3[i].n + k4[i].n);
        net[i].m = y0[i].m + dt/6*(k1[i].m + 2*k2[i].m + 2*k3[i].m + k4[i].m);
        net[i].h = y0[i].h + dt/6*(k1[i].h + 2*k2[i].h + 2*k3[i].h + k4[i].h);
        net[i].r = y0[i].r + dt/6*(k1[i].r + 2*k2[i].r + 2*k3[i].r + k4[i].r);
    }
}


/* ------------------------ Registro de spikes --------------------- */

// Chamada após cada passo: detecta V cruzando v_th subindo entre t e t+dt
// e grava o instante (interpolado linearmente) se já passou do transiente.
void record_spikes(const double vold[N], double t) {
    for (int i = 0; i < N; i++) {
        double vnew = net[i].v;
        if (vold[i] < v_th && vnew >= v_th) {
            double tc = t + dt * (v_th - vold[i]) / (vnew - vold[i]);
            if (tc < t_trans) continue;
            if (n_out[i] < MAX_OUT) spk[i][n_out[i]++] = tc;
            else overflow = 1;
        }
    }
}


/* ---------------------------- Métricas --------------------------- */

// <F>: spikes por neurônio por segundo, na janela de medição
double compute_F(void) {
    long total = 0;
    for (int i = 0; i < N; i++) total += n_out[i];
    double T_med = (tlim - t_trans) / 1000.0;   // em segundos
    return total / (N * T_med);
}

// CV: calculado por neurônio (σ_ISI/<ISI>) e depois promediado
// sobre os neurônios com pelo menos 3 spikes (2 ISIs).
// O artigo não especifica se agrupa os ISIs ou faz média por neurônio.
double compute_CV(int *n_valid) {
    double soma_cv = 0.0;
    int cont = 0;
    for (int i = 0; i < N; i++) {
        int ns = n_out[i];
        if (ns < 3) continue;
        double m = 0.0, m2 = 0.0;
        for (int k = 0; k < ns - 1; k++) {
            double isi = spk[i][k+1] - spk[i][k];
            m  += isi;
            m2 += isi * isi;
        }
        int nisi = ns - 1;
        m  /= nisi;
        m2 /= nisi;
        double var = m2 - m*m;
        if (var < 0) var = 0;
        soma_cv += sqrt(var) / m;
        cont++;
    }
    *n_valid = cont;
    return cont > 0 ? soma_cv / cont : NAN;
}

// <R>: média temporal do parâmetro de ordem de Kuramoto (Eq. 18),
// com a fase de cada neurônio dada pela Eq. 19.
// A fase só é definida entre o primeiro e o último spike de cada neurônio;
// por isso R é calculado no intervalo em que TODOS os neurônios com
// pelo menos 2 spikes têm fase definida. Neurônios com < 2 spikes ficam
// de fora (n_valid informa quantos entraram).
// grupo: 0 = todos, 1 = só estimulados, 2 = só não estimulados
static int no_grupo(int i, int grupo) {
    return grupo == 0 || (grupo == 1 && estimulado[i]) || (grupo == 2 && !estimulado[i]);
}

double compute_R(int *n_valid, int grupo) {
    double t_ini = -INFINITY, t_fim = INFINITY;
    int cont = 0;
    for (int i = 0; i < N; i++) {
        if (n_out[i] < 2 || !no_grupo(i, grupo)) continue;
        if (spk[i][0] > t_ini)            t_ini = spk[i][0];
        if (spk[i][n_out[i]-1] < t_fim)   t_fim = spk[i][n_out[i]-1];
        cont++;
    }
    *n_valid = cont;
    if (cont < 2 || t_fim <= t_ini) return NAN;

    int idx[N];                       // spike k tal que spk[k] <= t < spk[k+1]
    for (int i = 0; i < N; i++) idx[i] = 0;

    double soma_R = 0.0;
    long amostras = 0;
    for (double t = t_ini; t < t_fim; t += dt_R) {
        double sc = 0.0, ss = 0.0;
        for (int i = 0; i < N; i++) {
            if (n_out[i] < 2 || !no_grupo(i, grupo)) continue;
            while (idx[i] < n_out[i] - 2 && spk[i][idx[i]+1] <= t) idx[i]++;
            double ta = spk[i][idx[i]], tb = spk[i][idx[i]+1];
            double phi = 2.0 * M_PI * (t - ta) / (tb - ta);   // o termo 2πk some em e^{iφ}
            sc += cos(phi);
            ss += sin(phi);
        }
        soma_R += sqrt(sc*sc + ss*ss) / cont;
        amostras++;
    }
    return soma_R / amostras;
}

/* ------------------------ Saída em arquivo ----------------------- */

#define RESULTS_FILE "resultados_redes.csv"  // separado do CSV do neuro.c (colunas diferentes)
#define RASTER_DIR   "rasters"               // um arquivo por execução, se pedido
#define REDE_DIR     "redes"                 // lista de ligações, se pedido

// Acrescenta uma linha ao CSV de resultados. Escreve o cabeçalho se o
// arquivo estiver vazio. O flock garante que execuções simultâneas
// (xargs -P, GNU parallel) não misturem linhas nem dupliquem o cabeçalho.
void save_results(long seed, double R, double CV, double F, int nR, int nCV,
                  double R_est, double R_nao, int nR_est, int nR_nao) {
    FILE *f = fopen(RESULTS_FILE, "a");
    if (!f) { perror(RESULTS_FILE); return; }
    flock(fileno(f), LOCK_EX);

    fseek(f, 0, SEEK_END);
    if (ftell(f) == 0)
        fprintf(f, "rede,N,hubs,gExt,eps,nu,seed,tlim,R,CV,F,n_R,n_CV,"
                   "R_estim,R_nao_estim,n_R_estim,n_R_nao_estim\n");
    fprintf(f, "%s,%d,%d,%.4f,%.6f,%.3f,%ld,%.0f,%.5f,%.5f,%.3f,%d,%d,%.5f,%.5f,%d,%d\n",
            tipo_rede, N, n_hubs, gExt, eps, nu, seed, tlim, R, CV, F, nR, nCV,
            R_est, R_nao, nR_est, nR_nao);

    fflush(f);
    flock(fileno(f), LOCK_UN);
    fclose(f);
}

// Grava os spikes em formato "neurônio tempo" para raster plots (Fig. 4).
// O nome do arquivo identifica os parâmetros da execução.
void save_raster(long seed) {
    mkdir(RASTER_DIR, 0755);            // ignora o erro se a pasta já existe
    char nome[256];
    snprintf(nome, sizeof nome, "%s/raster_%s_N%d_h%d_g%.3f_e%.4f_nu%.2f_s%ld.txt",
             RASTER_DIR, tipo_rede, N, n_hubs, gExt, eps, nu, seed);
    FILE *f = fopen(nome, "w");
    if (!f) { perror(nome); return; }
    fprintf(f, "# rede=%s N=%d hubs=%d gExt=%g eps=%g nu=%g seed=%ld tlim=%g t_trans=%g\n",
            tipo_rede, N, n_hubs, gExt, eps, nu, seed, tlim, t_trans);
    fprintf(f, "# estimulados:");
    for (int i = 0; i < N; i++) if (estimulado[i]) fprintf(f, " %d", i);
    fprintf(f, "\n");
    fprintf(f, "# neuronio tempo_ms\n");
    for (int i = 0; i < N; i++)
        for (int k = 0; k < n_out[i]; k++)
            fprintf(f, "%d %.4f\n", i, spk[i][k]);
    fclose(f);
}

// Grava a rede usada: uma linha por neurônio com graus (e fitness, no BB),
// seguida da lista de ligações "j i" (j envia para i). Serve para conferir
// a distribuição de graus e, nas próximas etapas, para escolher os hubs.
void save_network(long seed) {
    mkdir(REDE_DIR, 0755);
    char nome[256];
    snprintf(nome, sizeof nome, "%s/rede_%s_N%d_h%d_s%ld.txt", REDE_DIR, tipo_rede, N, n_hubs, seed);
    FILE *f = fopen(nome, "w");
    if (!f) { perror(nome); return; }
    int kin[N], kout[N];
    degrees(kin, kout);
    fprintf(f, "# rede=%s N=%d seed=%ld K_MED=%d\n", tipo_rede, N, seed, K_MED);
    fprintf(f, "# NOS: neuronio grau_entrada grau_saida fitness estimulado\n");
    for (int i = 0; i < N; i++)
        fprintf(f, "N %d %d %d %.5f %d\n", i, kin[i], kout[i],
                strcmp(tipo_rede, "bianconi") ? 1.0 : fitness[i], estimulado[i]);
    fprintf(f, "# LIGACOES: pre pos (a[pos][pre] = 1)\n");
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (a[i][j]) fprintf(f, "E %d %d\n", j, i);
    fclose(f);
}


/* ------------------------------ main ----------------------------- */
//
// Uso:  ./neuro_redes gExt eps seed [nu] [tlim_ms] [salvar: 0 ou 1] [rede] [n_hubs]
//
//   rede:   random (padrão) | barabasi | bianconi
//   n_hubs: só para barabasi/bianconi (obrigatório nelas): quantos neurônios
//           de maior grau recebem o trem de Poisson; os demais têm gExt = 0.
//           Na rede random, todos são estimulados (como no artigo).
//
//   ./neuro_redes 0.1 0.1 1                            rede aleatória, 11 s
//   ./neuro_redes 0.1 0.1 1 1.0 11000 0 barabasi 10    BA, estimula os 10 maiores hubs
//   ./neuro_redes 0.1 0.1 1 1.0 11000 1 bianconi 5     BB, 5 hubs, + raster + rede
//
// Saída:
//   - resultados_redes.csv : uma linha acrescentada por execução
//         rede,N,hubs,gExt,eps,nu,seed,tlim,R,CV,F,n_R,n_CV,
//         R_estim,R_nao_estim,n_R_estim,n_R_nao_estim
//     (R_estim / R_nao_estim: R calculado só entre estimulados / não estimulados)
//   - rasters/raster_<rede>_N.._g.._e.._nu.._s...txt   (se salvar = 1)
//   - redes/rede_<rede>_N.._s...txt                     (se salvar = 1)
//   - resumo da rede (grau médio, mínimo, máximo) no stderr
//
// N é fixado na compilação (padrão 100):  gcc -O2 -DN=500 neuro_redes.c -o neuro_redes -lm
//
int main(int argc, char **argv){
  if (argc < 4) {
    fprintf(stderr, "uso: %s gExt eps seed [nu] [tlim_ms] [salvar: 0 ou 1] "
                    "[rede: random|barabasi|bianconi] [n_hubs]\n", argv[0]);
    return 1;
  }
  gExt = atof(argv[1]);
  eps  = atof(argv[2]);
  long seed = atol(argv[3]);
  if (argc > 4) nu   = atof(argv[4]);
  if (argc > 5) tlim = atof(argv[5]);
  int salvar = (argc > 6) ? atoi(argv[6]) : 0;
  if (argc > 7) {
    if (!net_type_valid(argv[7])) {
      fprintf(stderr, "erro: rede '%s' desconhecida (use random, barabasi ou bianconi)\n", argv[7]);
      return 1;
    }
    snprintf(tipo_rede, sizeof tipo_rede, "%s", argv[7]);
  }
  if (argc > 8) n_hubs = atoi(argv[8]);

  if (!strcmp(tipo_rede, "random")) {
    if (n_hubs >= 0 && n_hubs != N)
      fprintf(stderr, "aviso: rede random estimula todos os neuronios; n_hubs = %d ignorado\n", n_hubs);
    n_hubs = N;
  } else if (n_hubs < 0 || n_hubs > N) {
    fprintf(stderr, "erro: na rede %s informe o numero de hubs estimulados (8o argumento, 0 a %d)\n",
            tipo_rede, N);
    return 1;
  }

  if (tlim <= t_trans) {
    fprintf(stderr, "erro: tlim (%g ms) deve ser maior que o transiente (%g ms)\n",
            tlim, t_trans);
    return 1;
  }
  if (strcmp(tipo_rede, "random") && N <= K_MED / 2 + 1) {
    fprintf(stderr, "erro: N = %d pequeno demais para o crescimento preferencial\n", N);
    return 1;
  }

  if (nu * tlim > 0.9 * MAX_SPK)
    fprintf(stderr, "aviso: nu*tlim = %.0f se aproxima de MAX_SPK = %d; aumente MAX_SPK\n",
            nu * tlim, MAX_SPK);

  srand48(seed);
  // Inicialização da matriz de conexão
  build_network(tipo_rede);
  network_summary();
  select_stimulated();

  // Condições iniciais (artigo: V em [-80,0] mV; n, m, h em [0,1]).
  // O artigo não informa r(0); usamos r = 0 (nenhuma sinapse ativa).
  for (int i = 0; i < N; i++) {
    net[i].v = -80.0 + 80.0 * uniform();
    net[i].n = uniform();
    net[i].m = uniform();
    net[i].h = uniform();
    net[i].r = 0.0;
    if (estimulado[i]) generate_train(&net[i], nu, tlim);
    else { net[i].n_spk = 0; net[i].k0 = 0; }    // sem entrada externa
    if (net[i].n_spk == MAX_SPK)
      fprintf(stderr, "aviso: trem do neurônio %d truncado em MAX_SPK\n", i);
    n_out[i] = 0;
  }

  // Integração
  long nsteps = (long)(tlim / dt + 0.5);
  double vold[N];
  for (long st = 0; st < nsteps; st++) {
    double t = t0 + st * dt;
    for (int i = 0; i < N; i++) vold[i] = net[i].v;
    rk4_step(t);
    record_spikes(vold, t);
  }

  if (overflow) fprintf(stderr, "aviso: algum neurônio excedeu MAX_OUT spikes\n");

  // Métricas
  int nR, nCV;
  double R  = compute_R(&nR, 0);
  int nR_est, nR_nao;
  double R_est = compute_R(&nR_est, 1);
  double R_nao = compute_R(&nR_nao, 2);
  double CV = compute_CV(&nCV);
  double F  = compute_F();

  printf("%s,%d,%d,%.4f,%.6f,%.3f,%ld,%.0f,%.5f,%.5f,%.3f,%d,%d,%.5f,%.5f,%d,%d\n",
         tipo_rede, N, n_hubs, gExt, eps, nu, seed, tlim, R, CV, F, nR, nCV,
         R_est, R_nao, nR_est, nR_nao);

  save_results(seed, R, CV, F, nR, nCV, R_est, R_nao, nR_est, nR_nao);
  if (salvar) { save_raster(seed); save_network(seed); }

  return 0;
}
