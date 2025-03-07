
options(scipen = 999)

library(tidyverse)
library(readxl)
library(geosphere)



cns_inscricao <- read_excel("01_dados/edital_maismedicos_cns.xlsx") |> 
  select(cod_ibge, nome, resultado, uf) |> 
  rename(cns = resultado) |> 
  rename(municipio_destino = cod_ibge) |> 
  rename(uf_destino = uf) |> 
  mutate(cns = as.numeric(cns))


maismedicos_completo <- read_csv("01_dados/maismedicos_completo.csv")


base_tratada <- 
  maismedicos_completo |> 
  left_join(cns_inscricao, by = c("CNS_PROF" = "cns")) |> 
  mutate(COMPETEN = as.character(COMPETEN)) |> 
  mutate(ano = substr(COMPETEN, 1, 4)) |> 
  filter(COMPETEN == '201906' | COMPETEN == '202406') |> 
  mutate(mesma_cidade = if_else((municipio_destino == CODUFMUN), "Mesma cidade", "Outra cidade")) |> 
  mutate(mesmo_estado = if_else((uf_destino == uf_sigla), "Mesmo estado", "Outro estado")) |> 
  mutate(migracao = case_when(
    # Do Norte para outras regiões
    (uf_destino %in% c("AM", "AP", "TO", "AC", "RR", "RO", "PA")) & 
      uf_sigla %in% c("SC", "RS", "PR") ~ "Saiu do norte e foi para o sul",
    (uf_destino %in% c("AM", "AP", "TO", "AC", "RR", "RO", "PA")) & 
      uf_sigla %in% c("SP", "RJ", "ES", "MG") ~ "Saiu do norte e foi para o sudeste",
    (uf_destino %in% c("AM", "AP", "TO", "AC", "RR", "RO", "PA")) & 
      uf_sigla %in% c("GO", "MT", "MS", "DF") ~ "Saiu do norte e foi para o centro-oeste",
    (uf_destino %in% c("AM", "AP", "TO", "AC", "RR", "RO", "PA")) & 
      uf_sigla %in% c("AM", "TO", "AP", "RR", "PA", "RO", "AC") ~ "Permaneceu no norte",
    (uf_destino %in% c("AM", "AP", "TO", "AC", "RR", "RO", "PA")) & 
      uf_sigla %in% c("BA", "AL", "RN", "SE", "PI", "PB", "CE", "MA", "PE") ~ "Saiu do norte e foi para o nordeste",
    
    # Do Nordeste para outras regiões
    (uf_destino %in% c("BA", "AL", "RN", "SE", "PI", "PB", "CE", "MA", "PE")) & 
      uf_sigla %in% c("SC", "RS", "PR") ~ "Saiu do nordeste para o sul",
    (uf_destino %in% c("BA", "AL", "RN", "SE", "PI", "PB", "CE", "MA", "PE")) & 
      uf_sigla %in% c("SP", "RJ", "ES", "MG") ~ "Saiu do nordeste e foi para o sudeste",
    (uf_destino %in% c("BA", "AL", "RN", "SE", "PI", "PB", "CE", "MA", "PE")) & 
      uf_sigla %in% c("GO", "MT", "MS", "DF") ~ "Saiu do nordeste e foi para o centro-oeste",
    (uf_destino %in% c("BA", "AL", "RN", "SE", "PI", "PB", "CE", "MA", "PE")) & 
      uf_sigla %in% c("BA", "AL", "RN", "SE", "PI", "PB", "CE", "MA", "PE") ~ "Permaneceu no nordeste",
    (uf_destino %in% c("BA", "AL", "RN", "SE", "PI", "PB", "CE", "MA", "PE")) & 
      uf_sigla %in% c("AM", "AP", "TO", "AC", "RR", "RO", "PA") ~ "Saiu do nordeste e foi para o norte",
    
    # Do Centro-Oeste para outras regiões
    (uf_destino %in% c("GO", "MT", "MS", "DF")) & 
      uf_sigla %in% c("SC", "RS", "PR") ~ "Saiu do centro-oeste e foi para o sul",
    (uf_destino %in% c("GO", "MT", "MS", "DF")) & 
      uf_sigla %in% c("SP", "RJ", "ES", "MG") ~ "Saiu do centro-oeste e foi para o sudeste",
    (uf_destino %in% c("GO", "MT", "MS", "DF")) & 
      uf_sigla %in% c("BA", "AL", "RN", "SE", "PI", "PB", "CE", "MA", "PE") ~ "Saiu do centro-oeste e foi para o nordeste",
    (uf_destino %in% c("GO", "MT", "MS", "DF")) & 
      uf_sigla %in% c("AM", "AP", "TO", "AC", "RR", "RO", "PA") ~ "Saiu do centro-oeste e foi para o norte",
    (uf_destino %in% c("GO", "MT", "MS", "DF")) & 
      uf_sigla %in% c("GO", "MT", "MS", "DF") ~ "Permaneceu no centro-oeste",
    
    # Do Sul para outras regiões
    (uf_destino %in% c("SC", "RS", "PR")) & 
      uf_sigla %in% c("SP", "RJ", "ES", "MG") ~ "Saiu do sul e foi para o sudeste",
    (uf_destino %in% c("SC", "RS", "PR")) & 
      uf_sigla %in% c("GO", "MT", "MS", "DF") ~ "Saiu do sul e foi para o centro-oeste",
    (uf_destino %in% c("SC", "RS", "PR")) & 
      uf_sigla %in% c("AM", "AP", "TO", "AC", "RR", "RO", "PA") ~ "Saiu do sul e foi para o norte",
    (uf_destino %in% c("SC", "RS", "PR")) & 
      uf_sigla %in% c("BA", "AL", "RN", "SE", "PI", "PB", "CE", "MA", "PE") ~ "Saiu do sul e foi para o nordeste",
    (uf_destino %in% c("SC", "RS", "PR")) & 
      uf_sigla %in% c("SC", "RS", "PR") ~ "Permaneceu no sul",
    
    # Do Sudeste para outras regiões
    (uf_destino %in% c("SP", "RJ", "ES", "MG")) & 
      uf_sigla %in% c("GO", "MT", "MS", "DF") ~ "Saiu do sudeste e foi para o centro-oeste",
    (uf_destino %in% c("SP", "RJ", "ES", "MG")) & 
      uf_sigla %in% c("AM", "AP", "TO", "AC", "RR", "RO", "PA") ~ "Saiu do sudeste e foi para o norte",
    (uf_destino %in% c("SP", "RJ", "ES", "MG")) & 
      uf_sigla %in% c("SC", "RS", "PR") ~ "Saiu do sudeste e foi para o sul",
    (uf_destino %in% c("SP", "RJ", "ES", "MG")) & 
      uf_sigla %in% c("BA", "AL", "RN", "SE", "PI", "PB", "CE", "MA", "PE") ~ "Saiu do sudeste e foi para o nordeste",
    (uf_destino %in% c("SP", "RJ", "ES", "MG")) & 
      uf_sigla %in% c("SP", "RJ", "ES", "MG") ~ "Permaneceu no sudeste",
    
    TRUE ~ "outros casos"
  ))

base_tratada <- base_tratada %>% 
  mutate(regiao_brasil_destino = case_when(
    (uf_destino %in% c("AM", "AP", "TO", "AC", "RR", "RO", "PA")) ~ "Norte",
    (uf_destino %in% c("BA", "AL", "RN", "SE", "PI", "PB", "CE", "MA", "PE")) ~ "Nordeste",
    (uf_destino %in% c("GO", "MT", "MS", "DF")) ~ "Centro-oeste",
    (uf_destino %in% c("SC", "RS", "PR")) ~ "Sul",
    (uf_destino %in% c("SP", "RJ", "ES", "MG")) ~ "Sudeste",
    TRUE ~ "Outro"))

base_tratada <- base_tratada %>% 
  mutate(regiao_brasil_exercicio = case_when(
    (uf_sigla %in% c("AM", "AP", "TO", "AC", "RR", "RO", "PA")) ~ "Norte",
    (uf_sigla %in% c("BA", "AL", "RN", "SE", "PI", "PB", "CE", "MA", "PE")) ~ "Nordeste",
    (uf_sigla %in% c("GO", "MT", "MS", "DF")) ~ "Centro-oeste",
    (uf_sigla %in% c("SC", "RS", "PR")) ~ "Sul",
    (uf_sigla %in% c("SP", "RJ", "ES", "MG")) ~ "Sudeste",
    TRUE ~ "Outro"))




#Contagem de profissionais na competência 201906
compet2019 <- base_tratada |>
  filter(ano == 2019) |> 
  group_by(CNS_PROF) |> 
  count()

compet2019$CNS_PROF <- as.character(compet2019$CNS_PROF)
base_tratada$CNS_PROF <- as.character(base_tratada$CNS_PROF)

#Contagem de profissionais na competência 202406
compet2024 <- base_tratada |>
  filter(ano == 2024) |> 
  group_by(CNS_PROF) |> 
  count()


#Filtrando a base para manter apenas os profissionais que estão presentes em ambas competências
base_tratada <- base_tratada |> 
  filter(CNS_PROF %in% intersect(compet2019$CNS_PROF, compet2024$CNS_PROF))


#Contagem de profissionais chamados por UF
tabela_uf_destino <- base_tratada %>% 
  filter(ano == 2019) %>%
  group_by(uf_destino) %>%
  summarise(cont_destino = n_distinct(CNS_PROF))

tabela_uf_destino %>%
  ungroup() %>% 
  summarise(total = sum(cont_destino))

#Contagem de profissionais por UF em exercicio em 2024
tabela_uf_exercicio <- base_tratada %>% 
  filter(ano == 2024) %>%
  group_by(uf_sigla) %>%
  summarise(cont_exercicio = n_distinct(CNS_PROF))

tabela_uf_exercicio %>%
  ungroup() %>% 
  summarise(total = sum(cont_exercicio))

#Comparativo de profissionais por UF de destino e de exercicio
comparativo_uf <- tabela_uf_destino |> 
  left_join(tabela_uf_exercicio, by = c("uf_destino" = "uf_sigla")) |> 
  mutate(percentual = cont_exercicio/(cont_destino))

#Contagem de profissionais chamados por região do Brasil
tabela_regiao_destino <- base_tratada %>% 
  filter(ano == 2019) %>%
  group_by(regiao_brasil_destino) %>%
  summarise(cont_regiao_destino = n_distinct(CNS_PROF))

tabela_regiao_destino %>%
  ungroup() %>% 
  summarise(total = sum(cont_regiao_destino))

#Contagem de profissionais em exercicio em 2024 por região do Brasil
tabela_regiao_exercicio<- base_tratada %>% 
  filter(ano == 2024) %>%
  group_by(regiao_brasil_exercicio) %>%
  summarise(cont_regiao_exercicio = n_distinct(CNS_PROF))

tabela_regiao_exercicio %>%
  ungroup() %>% 
  summarise(total = sum(cont_regiao_exercicio))

#Comparativo de profissionais por região do Brasul de destino e de exercicio
comparativo_regiao <- tabela_regiao_destino |> 
  left_join(tabela_regiao_exercicio, by = c("regiao_brasil_destino" = "regiao_brasil_exercicio"))
  

tipologia_municipal <- read_xlsx("01_dados/tipologia_municipal_rural_urbano.xlsx") |> 
  mutate(CD_GCMUN = substr(CD_GCMUN,1,6))


#Agregando tipologia municipal de onde o profissional atua
base_tratada$CODUFMUN <- as.character(base_tratada$CODUFMUN)

base_tratada <- base_tratada |> 
  left_join(tipologia_municipal, by = c("CODUFMUN" = "CD_GCMUN")) |> 
  select(-NM_UF, -SIG_UF, -NM_MUN) |> 
  rename(tipo_exercicio = TIPO)

base_tratada$tipo_exercicio <- gsub("Rural", "Rural ",base_tratada$tipo_exercicio)
base_tratada$tipo_exercicio <- gsub("Intermediario", "Intermediario ",base_tratada$tipo_exercicio)

#Agregando tipologia municipal de onde o profissional foi chamado
base_tratada$municipio_destino <- as.character(base_tratada$municipio_destino)
base_tratada <- base_tratada |> 
  left_join(tipologia_municipal, by = c("municipio_destino" = "CD_GCMUN")) |> 
  select(-NM_UF, -SIG_UF, -NM_MUN) |> 
  rename(tipo_destino = TIPO)

base_tratada$tipo_destino <- gsub("Rural", "Rural ",base_tratada$tipo_destino)
base_tratada$tipo_destino <- gsub("Intermediario", "Intermediario ",base_tratada$tipo_destino)

base_tratada <- base_tratada |> 
  mutate(migracao_tipo = ifelse((tipo_destino == tipo_exercicio), paste("Permaneceu no", tipo_destino, sep = " "),
                                paste("Saiu do", tipo_destino, "e foi para o", tipo_exercicio)))

teste <- base_tratada |> 
  filter(ano == 2024) |> 
  group_by(migracao_tipo) |> 
  count() |> 
  mutate(percentual = n/ sum(n) )

teste <- base_tratada |> 
  filter(ano == 2024) |> 
  group_by(migracao_tipo) |> 
  count() |> 
  ungroup() |> 
  mutate(percentual = round((n / sum(n))*100, 2))

teste |> 
  ggplot(aes(x = reorder(migracao_tipo, percentual), y = percentual)) + geom_col() + 
  coord_flip() + theme_minimal()


regioes_influencia <- read_xlsx("01_dados/regioes_influencia.xlsx") |> 
  mutate(codmun = substr(codmun,1,6)) |> 
  select(codmun, regiao_influencia)

#Agregando a região de influência do municipio que o profissional foi chamado
base_tratada <- base_tratada |> 
  left_join(regioes_influencia, by = c("municipio_destino" = "codmun")) |> 
  rename(regiao_influencia_destino = regiao_influencia)

#Agregando a região de influência do municipio que o profissional atua
base_tratada <- base_tratada |> 
  left_join(regioes_influencia, by = c("CODUFMUN" = "codmun")) |> 
  rename(regiao_influencia_exercicio = regiao_influencia)

base_tratada <- base_tratada %>% 
  mutate(migraçao_regiao_influencia = ifelse(regiao_influencia_destino == regiao_influencia_exercicio,
                                             "Permaneceu na mesma região", "Migrou para outra região"))

retencao_regiao_influencia <- base_tratada %>% 
  filter(ano == 2024) %>% 
  group_by(uf_destino, migraçao_regiao_influencia) %>% 
  count()

retencao_regiao_influencia <- retencao_regiao_influencia |> 
  pivot_wider(names_from = migraçao_regiao_influencia,  values_from = n) |> 
  rename(migrou_outra_regiao = `Migrou para outra região`) |> 
  rename(permaneceu_regiao = `Permaneceu na mesma região`)

teste <- retencao_regiao_influencia |> 
  mutate(retencao = permaneceu_regiao/(permaneceu_regiao + migrou_outra_regiao))


teste %>% 
  ggplot(aes(x = n, y = reorder(uf_destino, n), color = migraçao_regiao_influencia)) +
           geom_col()
