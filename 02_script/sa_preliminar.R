options(scipen = 999)

install.packages("geosphere")
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
  left_join(cns_inscricao, by = c("CNS_PROF"="cns")) |> 
  mutate(COMPETEN = as.character(COMPETEN)) |> 
  mutate(ano = substr(COMPETEN, 1, 4)) |> 
  mutate(mesma_cidade = if_else((municipio_destino == CODUFMUN), "Mesma cidade","Outra cidade")) |> 
  mutate(mesmo_estado = if_else((uf_destino == uf_sigla), "Mesmo estado", "Outro estado")) |> 
  group_by(CNS_PROF, COMPETEN) |> 
  mutate(vinculo = if_else(n() > 1, "Multivinculo", "Vinculo unico")) |> 
  ungroup()



base_tratada |> 
  filter(ano == '2024') |> 
  group_by(NOMEPROF, mesmo_estado) |> 
  count() |> 
  ungroup() |> 
  group_by(mesmo_estado) |> 
  count() |> 
  ggplot(aes(x = mesmo_estado, y = n)) + geom_col()
