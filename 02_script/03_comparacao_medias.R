options(scipen = 999)

library(tidyverse)
library(rstatix)
library(vcd)


df <- 
  read_csv("01_dados/dados resultantes/df_modelagem0303.csv")

df |> 
  group_by(churn) |> 
  count()


# idade -------------------------------------------------------------------

df |> 
  group_by(churn) |> 
  summarise(idade_media = mean(idade),
            idade_sd = sd(idade))

df |> 
  ggplot(aes(x = idade, fill = churn)) + geom_histogram()

shapiro.test(df$idade)
# se p < 0.05 nao é normal 

wilcox.test(idade ~ churn, data = df, exact = FALSE)


# Anos formacao -----------------------------------------------------------

df |> 
  group_by(churn) |> 
  summarise(mean(anos_formacao),
            sd(anos_formacao))


shapiro.test(df$anos_formacao)

wilcox.test(anos_formacao ~ churn, data = df, exact = FALSE)


# Quantidade média de vínculos de trabalho --------------------------------

df |> 
  group_by(churn) |> 
  summarise(mean(media_vinculos_mes),
            sd(media_vinculos_mes))


shapiro.test(df$media_vinculos_mes)

wilcox.test(media_vinculos_mes ~ churn, data = df, exact = FALSE)

# Quantidade média de enfermeiros --------------------------------

df |> 
  group_by(churn) |> 
  summarise(mean(m_enfermeiro, na.rm = TRUE),
            sd(m_enfermeiro, na.rm = TRUE))


shapiro.test(df$m_enfermeiro)

wilcox.test(m_enfermeiro ~ churn, data = df, exact = FALSE)


# Quantidade média de tec. de enfermagem --------------------------------

df |> 
  group_by(churn) |> 
  summarise(mean(m_tec_aux_enf, na.rm = TRUE),
            sd(m_tec_aux_enf, na.rm = TRUE))


shapiro.test(df$m_tec_aux_enf)

wilcox.test(m_tec_aux_enf ~ churn, data = df, exact = FALSE)


# Prorrogacao -------------------------------------------------------------

df |> 
  group_by(churn) |> 
  count(Prorrogado)


# Criar a tabela de contingência
tabela <- table(df$churn, 
                df$Prorrogado)

# Aplicar o teste Qui-Quadrado
chisq.test(tabela)

assocstats(tabela)


# região de destino -------------------------------------------------------

df |> 
  group_by(churn) |> 
  count(regiao_destino)


# Criar a tabela de contingência
tabela <- table(df$churn, 
                df$regiao_destino)

chisq.test(tabela)

assocstats(tabela)


