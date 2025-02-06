#install.packages("wdman")
library(readxl)
library(tidyverse)
library(RSelenium)
library(wdman)
library(netstat)
library(writexl)


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
remdr$maxWindowSize()

# Inicializando o dataframe com uma coluna para resultados
edital$resultado <- NA  # Cria a coluna de resultado inicialmente com NA

# Loop atrav??s dos nomes na coluna "nome"
for (i in 1:nrow(edital)) {
  # Abrindo o site
  remdr$navigate('https://cnes.datasus.gov.br/pages/profissionais/consulta.jsp')
  
  # Realizando a pesquisa
  campo_pesquisa <- remdr$findElement(using = "css selector", value = "#pesquisaValue")
  campo_pesquisa$sendKeysToElement(list(edital$nome[i], key = 'enter'))
  
  # Aguardando o carregamento da p??gina
  Sys.sleep(2)  # Aguarde um momento para garantir que a p??gina carregue
  
  # Verificando se existe o elemento na segunda linha da tabela
  existe_segunda_linha <- tryCatch({
    remdr$findElement(using = "xpath", value = "/html/body/div[2]/main/div/div[2]/div/div[4]/table/tbody/tr[2]/td[1]")
  }, error = function(e) {
    NULL  # Retorna NULL se o elemento n??o for encontrado
  })
  
  # Se a segunda linha for encontrada, armazena NA e pula para o pr??ximo nome
  if (!is.null(existe_segunda_linha)) {
    edital$resultado[i] <- NA
    print(paste("Nome:", edital$nome[i], "- Encontrado mais de um resultado, armazenando como NA."))
    next  # Pula para o pr??ximo item do loop
  }
  
  # Tentando capturar o CNS da primeira linha com tryCatch para evitar erros se n??o for encontrado
  CNS <- tryCatch({
    remdr$findElement(using = "xpath", value = "/html/body/div[2]/main/div/div[2]/div/div[4]/table/tbody/tr/td[1]")
  }, error = function(e) {
    NULL  # Retorna NULL se ocorrer um erro
  })
  
  # Verifique se o elemento CNS foi encontrado
  if (!is.null(CNS)) {
    resultado <- CNS$getElementText()[[1]]  # Captura o texto do elemento
    edital$resultado[i] <- resultado
    print(paste("Nome:", edital$nome[i], "- Resultado encontrado:", resultado))
  } else {
    edital$resultado[i] <- NA  # Caso n??o haja resultados ou elemento n??o seja encontrado
    print(paste("Nome:", edital$nome[i], "- Resultado n??o encontrado."))
  }
}

write_xlsx(edital, "C:/Users/alefs/OneDrive/Documentos/GitHub/sa_maismedicos/dados/edital_maismedicos_cns.xlsx")

# Atribuindo ao dataframe
edital$resultado <- resultados_vector