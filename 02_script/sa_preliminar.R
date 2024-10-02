library(tidyverse)
library(readxl)

base <- read.csv("GitHub/sa_maismedicos/dados/maismedicos_completo.csv")


# Criando variável de localidade e selecionando as variáveis de interesse
tratamento <- base |> 
  mutate(local = paste(uf_sigla, municipio, sep = "-")) |> 
  select(local, NOMEPROF, CNS_PROF, COMPETEN) |> 
  mutate(CNS_PROF = as.character(CNS_PROF))

#Atribuindo formato data para competência e filtrando apenas as datas posteriores a janeiro/2019
tratamento2 <- tratamento %>%
  mutate(COMPETEN = as.Date(paste0(COMPETEN, "01"), format = "%Y%m%d")) %>%  # Convertendo para data
  arrange(COMPETEN) |> 
  filter(COMPETEN >= as.Date("2019-04-01"))

#Transformando os dados de longo para largo
dados_largo <- tratamento2 |> 
  pivot_wider(names_from = COMPETEN, values_from = local)

#Retirando os "NULL"
teste <- dados_largo |> 
  mutate(across(3:30, as.character)) |> 
  filter(if_all(2:67, ~ . != "NULL"))