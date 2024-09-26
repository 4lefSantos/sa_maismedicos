#install.packages("wdman")
library(readxl)
library(tidyverse)
library(RSelenium)
library(wdman)
library(netstat)


#Abrindo dataframe
edital <- read_excel("~/GitHub/sa_maismedicos/dados/edital_maismedicos.xlsx")

# Abrindo driver
binman::list_versions("chromedriver")
Sys.setlocale("LC_ALL", "C")
driver <- rsDriver(browser = "chrome",
                   chromever = "129.0.6668.70",
                   verbose = FALSE,
                   port = free_port())
remdr <- driver$client
remdr$maxWindowSize

# Abrindo site
remdr$navigate('https://cnes.datasus.gov.br/pages/profissionais/consulta.jsp')


#Realizando a pesquisa
campo_pesquisa <- remdr$findElement(using = "css selector", value = "#pesquisaValue")
campo_pesquisa$sendKeysToElement(list("ABIMAEL CRUZ NASCIMENTO", key = 'enter'))


#Retornando os valores
CNS <- remdr$findElement(using = "xpath", value = "//td[contains(@class, 'ng-binding')]")
length(CNS)
CNS$getElementText()

edital <- edital %>% sample_n(10)
#******************************************
  
# Abrindo driver
binman::list_versions("chromedriver")
Sys.setlocale("LC_ALL", "C")
driver <- rsDriver(browser = "chrome",
                   chromever = "129.0.6668.70",
                   verbose = FALSE,
                   port = free_port())
remdr <- driver$client
remdr$maxWindowSize()

# Inicializando o dataframe com uma coluna para resultados
edital$resultado <- NA  # Cria a coluna de resultado inicialmente com NA

# Loop através dos nomes na coluna "nome"
for (i in 1:nrow(edital)) {
  # Abrindo site
  remdr$navigate('https://cnes.datasus.gov.br/pages/profissionais/consulta.jsp')
  
  # Realizando a pesquisa
  campo_pesquisa <- remdr$findElement(using = "css selector", value = "#pesquisaValue")
  campo_pesquisa$sendKeysToElement(list(edital$nome[i], key = 'enter'))
  
  # Retornando os valores
  Sys.sleep(2)  # Aguarde um momento para garantir que a página carregue
  CNS <- remdr$findElements(using = "xpath", value = "//td[contains(@class, 'ng-binding')]")
  
  # Debug: verificar o conteúdo de CNS
  print(CNS)
  
  # Verifique se existem elementos encontrados
  if (length(CNS) > 0) {
    # Armazena o texto do primeiro elemento encontrado, por exemplo
    edital$resultado[i] <- unlist(lapply(CNS, function(x) x$getElementText()))[1]  # Captura o primeiro texto
    print(paste("Nome:", edital$nome[i], "- Resultado encontrado:", edital$resultado[i]))
  } else {
    edital$resultado[i] <- NA  # Caso não haja resultados
    print(paste("Nome:", edital$nome[i], "- Resultado não encontrado."))
  }
}

# Atribuindo ao dataframe
edital$resultado <- resultados_vector