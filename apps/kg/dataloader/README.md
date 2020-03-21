#Preprocess
Select partial relations from the whole set: drug_preprocess.py

 * entity entity2id
 * relation relation2id
 * triple the triple file in format head_id rel_id tail_id
 * selected_rels List of relaiton_id

It will output three files: new_entity2id, new_relation2id, new triples

```
python drug_preprocess.py --entity ~/kg/entity2id.tsv --relation ~/kg/relation2id.tsv --triple ~/kg/triples.tsv --selected_rels 6 7 8 16 17 19 33 47 48 34 57 59 60 68 69 64
```

# Split dataset
Split triples into train, valid and test, Note it only split specified relation_id into train, valid and test, other relations are all put into train. You should change L37 before run. 
```
preprocess.py --file ../drug/triples.tsv 
```

# Preprocess target drug and target into new id space
drug_id_remap.py will map drug and target into new id space according id_map and target_map
```
python drug_id_remap.py --src_drug ~/drug-protien/drug_target.tsv --id_map ../drug/entities.tsv --src_target ~/drug-protien/drug_target.tsv --target_map ../drug/entities.tsv
```

# Train
```
python3 train.py --model TransE_l2 --dataset udd --format udd --data_path drug --data_files entities.tsv relations.tsv triples.tsv valid.tsv test.tsv  --batch_size 1024 --neg_sample_size 256 --regularization_coef=2e-7 --hidden_dim 400 --gamma 12.0 --lr 0.1 --batch_size_eval 16 --valid --test -adv  --gpu 0 --max_step 128000 --neg_sample_size_valid 1000 --neg_sample_size_test 1000 --no_eval_filter --batch_size_eval 1000 --save_emb drug_model
```

#Inference
```
 python3 infer.py --model_name TransE_l2 --dataset udd --format udd --data_path drug --data_files entities.tsv relations.tsv triples.tsv valid.tsv test.tsv --hidden_dim 400 --gamma 12.0 --model_path drug_model --predict_head dataloader/drug_entity.tsv --predict_tail dataloader/target_entity.tsv 
```
