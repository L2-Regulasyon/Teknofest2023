cd src

# List of embedding models
declare -a EmbedModels=("dbmdz/bert-base-turkish-128k-uncased" "xlm-roberta-base" "xlm-roberta-large")

# Vector Stack Routine
for vector_model in tfidf fasttext
do
  for head_model in lgbm xgb catboost
  do
     python train_vector_stack.py -vector-model $vector_model -head-model $head_model --add-zoo
  done
done

# Embedding Stack Routine
for embedding_model in ${EmbedModels[@]}; do
  for head_model in lgbm xgb catboost svc
  do
     python train_embedding_stack.py -embedding-model-path $embedding_model -head-model $head_model --add-zoo
     python train_embedding_stack.py -embedding-model-path $embedding_model -head-model $head_model --retrain-embed-model --add-zoo
  done
done

# BERT Variant Routine
for embedding_model in ${EmbedModels[@]}; do
  python train_bert.py -model-path $embedding_model --add-zoo --cv
done

# Unbiased versions of the best BERT Variant
python train_bert.py -prevent-bias 1 --add-zoo --cv
python train_bert.py -prevent-bias 2 --add-zoo --cv