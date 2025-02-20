# Criar o vetor com aspas simples ao redor de cada elemento
vetor <- paste0("'", lista_cpf$CPF_PROF, "'")

# Exibir o vetor sem aspas duplas na saída
vetor <- paste(vetor, collapse = ",")

#É imporante salvar em txt e só depois copiar para a consulta
writeLines(vetor, "vetor_cpf_prof.txt")
