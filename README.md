# Master thesis repository Ole Magnus Gaalaas-Hansen Candidate nr: 1030

Welcome to my repository!

This repository is for the masters thesis "Foreign Experts in Autocratic Media: A Study of Modern Russian Propaganda Strategies" 

Here you will find the scripts used in the thesis and the final datasets the analyses are done on. 
It is strutured the following way: 

SCRIPTS: 
I have used both R and Python for this thesis. Python is mainly used for the compuational methods described in chapter 4 and 5 , which is the process of going from dataset of raw newspaper articles to the analysis ready dataset "expert_topic_sentiment_master_anonymised", although i prefer working in R it was not well suited the methods used in the thesis. While R is mainly used for the analysis. 
Here are an explanination of the folder structures and how to navigate the repistory, feel free to try to run scripts! 

The folder "Translation" provides the Python scripts for translation.
The first script is "Russian_roles_extraction" this is the dataset that uses keywords in russian to extract the relevant articles.
The second script "translate_experts_subset_yandex" is the script used for translating the relevant articles from Russian to English. If you decide to run that script i recommend you try the "translation_dashboard" script since it shows the real time progress and esitmated time left. the script took four days on my computer so i highly recommend the dashboard- 
The third script is "merge_shards" this is neccersary since the translation script runs on two CPUs (to save time) and creates two datasets. this is used for combingin them. 

The folder "Expert exctraction" contains the scripts for extracting experts and other metadata form the translated newspaper data.
"expert_exctraction_1999_2024" is the main script. It took my computer about a week to run. 
"expert_exctraction_izvestia" is the script used in the robustness test for Izvestia. 

The folder "topic modelling" provides the R script for topic modelling. 

The folder "Directed sentiment" proveds the scripts for the sentiment procedures. 
"03_dependency_directed_sentiment" is the main script. 
"03_dependency_directed_sentiment_dropped" is for the dropped experts robustness test 
"03_dependency_directed_sentiment_izvestia" is for Izvestia articles in the robustness test. 
"04a_validation_sentence_window" is the script for the LLM test of different sentence windows for directed sentiment"
"04b_validation_llm_judge 2" is the script for the LLM validation. 

The folder "Analysis and visualization" contains the script "analysis" which are used for the analysis. You should be able to run using the "expert_topic_sentiment_master_anonymised" dataset. 

DATA: 
"expert_topic_sentiment_master_anonymised" is the main dataset used for the analysis. 
The dataset is anonymized as per the recomodation from SIKT, none of those variables are needed for the analyses, but they were crucial for creating the variables used. 

I do not provide the rest of the datasets used for three reasons. 
1) The dataset provided contains all the relevant variables form the earlier stages (extraction, topic modelling and directed sentiment variables)
2) The raw dataset is the intelectual property of Yutkin. His reposity is cited in the thesis and you can download it from there. I do not know where the line of intelectual property goes in GitHub. I do not know of the translation is enough and i do not want my GitHub profile to be banned. (this has happend once before when i was less carefull)
3) Most of the dataset before the final dataset is mainly things done for purposes of creating the final one and are depenedent on the anaonmized varaibles so it would be quite redundant to provide them.





