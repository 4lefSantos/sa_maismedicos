
library(readxl)
library(tidyverse)
library(writexl)

edital_1 <- read_excel("01_dados/Editais/2019/Edital_11_maio_2019_primeira_fase_ciclo_18_resultado.xlsx") |> 
  mutate(fase = "1")

edital_2_preliminar <- read_excel("01_dados/Editais/2019/Edital_11_maio_2019_segunda_fase_ciclo_18_preliminar.xlsx")

edital_2_resultado <- read_excel("01_dados/Editais/2019/Edital_11_maio_2019_segunda_fase_ciclo_18_resultado.xlsx") |> 
  mutate(homolog = "Homologado") |> 
  select(NOME, homolog)

edital_2 <- edital_2_preliminar |> 
  left_join(edital_2_resultado, by = c("nome" = "NOME")) |> 
  filter(!is.na(homolog)) |> 
  mutate(fase = "2")

edital_2 <- edital_2 |> 
  select(-homolog)

edital_2$ibge_2 <- as.character(edital_2$ibge_2)
edital_2$ibge_3 <- as.character(edital_2$ibge_3)
edital_2$ibge_4 <- as.character(edital_2$ibge_4)



edital <- bind_rows(edital_1, edital_2) |> 
  select(-atendido)

write_xlsx(edital, "01_dados/Editais/2019/Edital_11_2019_resultado.xlsx")