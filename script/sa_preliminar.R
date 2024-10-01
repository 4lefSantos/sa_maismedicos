library(tidyverse)
library(readxl)

base <- read.csv("GitHub/sa_maismedicos/dados/maismedicos_completo.csv")

tratamento <- base |> 
  mutate(local = paste(uf_sigla, municipio, sep = "-")) |> 
  select(local, NOMEPROF, COMPETEN)

tratamento <- tratamento %>%
  mutate(COMPETEN = as.Date(paste0(COMPETEN, "01"), format = "%Y%m%d")) %>%  # Convertendo para data
  arrange(COMPETEN) |> 
  filter(COMPETEN >= as.Date("2019-03-01"))


dados_largo <- tratamento |> 
  pivot_wider(names_from = COMPETEN, values_from = local)

teste <- dados_largo |> 
  filter(if_all(2:67, ~ . != "NULL"))