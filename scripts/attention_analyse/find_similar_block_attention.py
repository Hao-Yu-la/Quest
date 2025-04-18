from transformers import AutoTokenizer
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from math import ceil
import json

MODEL_PATH = "/home/zhanghaoyu/models/Llama-3.1-8B-Instruct/"
DEVICE = torch.device("cuda:1")
DTYPE = torch.float16
NUM_LAYERS = 32
NUM_HEADS = 32
torch.set_default_dtype(DTYPE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

method = "hf"
token_budget = 512
topp = None
block_size = 16
topk = token_budget // block_size

if method == "quest":
  from quest import LlamaForCausalLM
  model = LlamaForCausalLM.from_pretrained(MODEL_PATH, device_map=DEVICE, torch_dtype=DTYPE, output_attentions=True)

  # Init Quest Controller
  model.quest_init(page_size=16, max_seq_len=16384, token_budget=token_budget, topp=topp)
else:
  from transformers import LlamaForCausalLM
  model = LlamaForCausalLM.from_pretrained(MODEL_PATH, device_map=DEVICE, torch_dtype=DTYPE, output_attentions=True)
  
prompt = [
    "In an animal kingdom, the lion is the king. One day, the lion announces a competition to choose the most hardworking animal. The turtle, rabbit, monkey, zebra, and giraffe all decide to participate. After a day of observation, the lion notices that all the animals are working hard, except for the rabbit, who is sleeping. So why does the lion choose the rabbit as the most hardworking animal?",
    "Summary the following passages: Passage 1:\nStarting in 1996, Alexa Internet has been donating their crawl data to the Internet Archive. Flowing in every day, these data are added to the Wayback Machine after an embargo period.\nPassage 2:\nImage copyright Getty Images Image caption Kalashnikov designed the AK-47 after being wounded fighting for the Red Army NEWLINE_CHAR NEWLINE_CHAR The inventor of the Kalashnikov assault rifle apparently wrote to the head of the Russian Orthodox Church before he died expressing fears he was morally responsible for the people it killed. NEWLINE_CHAR NEWLINE_CHAR Mikhail Kalashnikov, who died last month aged 94, wrote a long emotional letter to Patriarch Kirill in May 2012, church officials say. NEWLINE_CHAR NEWLINE_CHAR He said he was suffering \"spiritual pain\" over the many deaths it caused. NEWLINE_CHAR NEWLINE_CHAR Kalashnikov had previously refused to accept responsibility for those killed. NEWLINE_CHAR NEWLINE_CHAR 'Devilish desires' NEWLINE_CHAR NEWLINE_CHAR Analysis The letter published by Izvestia provides a fascinating insight into the mind of the man who created Russia's most famous weapon. Mikhail Kalashnikov spent his career designing and perfecting assault rifles. More than 100 million Kalashnikovs have been sold worldwide. The gun brought Kalashnikov fame and a string of awards. But his letter to the Patriarch suggests that, towards the end of his life, Kalashnikov felt a degree of guilt - or \"spiritual pain\" as he puts it - for having invented a killing machine. It's unclear, though, how much of this he wrote himself. Izvestia quotes Kalashnikov's daughter, Elena, as saying she believes a priest helped her father compose the letter. NEWLINE_CHAR NEWLINE_CHAR But in a letter, published in Russia's pro-Kremlin newspaper Izvestia, he wrote: \"My spiritual pain is unbearable. NEWLINE_CHAR NEWLINE_CHAR \"I keep having the same unsolved question: if my rifle claimed people's lives, then can it be that I... a Christian and an Orthodox believer, was to blame for their deaths?\" he asked. NEWLINE_CHAR NEWLINE_CHAR \"The longer I live,\" he continued, \"the more this question drills itself into my brain and the more I wonder why the Lord allowed man to have the devilish desires of envy, greed and aggression\". NEWLINE_CHAR NEWLINE_CHAR The letter is typed on Kalashnikov's personal writing paper, and is signed with a wavering hand by the man who describes himself as \"a slave of God, the designer Mikhail Kalashnikov\". NEWLINE_CHAR NEWLINE_CHAR The Kalashnikov, or AK-47, is one of the world's most familiar and widely used weapons. NEWLINE_CHAR NEWLINE_CHAR Its comparative simplicity made it cheap to manufacture, as well as reliable and easy to maintain. NEWLINE_CHAR NEWLINE_CHAR It is thought that more than 100 million Kalashnikov rifles have been sold worldwide. NEWLINE_CHAR NEWLINE_CHAR Kalashnikov refused to accept responsibility for the many people killed by his weapon, blaming the policies of other countries that acquired it. NEWLINE_CHAR NEWLINE_CHAR Image copyright Reuters Image caption Russian President Vladimir Putin attended Kalashnikov's funeral in December NEWLINE_CHAR NEWLINE_CHAR However, pride in his invention was tempered with sadness at its use by criminals and child soldiers. NEWLINE_CHAR NEWLINE_CHAR \"It is painful for me to see when criminal elements of all kinds fire from my weapon,\" Kalashnikov said in 2008. NEWLINE_CHAR NEWLINE_CHAR He designed this rifle to defend his country, not so terrorists could use it in Saudi Arabia Cyril Alexander Volkov, Press secretary for Russian Patriarch Kirill NEWLINE_CHAR NEWLINE_CHAR Defend his country NEWLINE_CHAR NEWLINE_CHAR In his letter to Patriarch Kirill, Kalashnikov said that he first went into a church at the age of 91 and was later baptised. NEWLINE_CHAR NEWLINE_CHAR The BBC's Steve Rosenberg in Moscow says it is unclear how much of it he wrote himself. Izvestia quotes Kalashnikov's daughter, Elena, as saying she believes a priest helped her father compose the letter. NEWLINE_CHAR NEWLINE_CHAR The press secretary for the Russian Patriarch, Cyril Alexander Volkov, told the paper the religious leader had received Kalashnikov's letter and had written a reply. NEWLINE_CHAR NEWLINE_CHAR \"The Church has a very definite position: when weapons serve to protect the Fatherland, the Church supports both its creators and the soldiers who use it,\" Mr Volkov was quoted as saying. NEWLINE_CHAR NEWLINE_CHAR \"He designed this rifle to defend his country, not so terrorists could use it in Saudi Arabia.\" NEWLINE_CHAR NEWLINE_CHAR Kalashnikov received many Russian state honours, including the Order of Lenin and the Hero of Socialist Labour, but made little money from his gun. NEWLINE_CHAR NEWLINE_CHAR He died on 23 December after being admitted to hospital a month earlier with internal bleeding.\nPassage 3:\nIn this Tuesday, Nov. 10, 2009 file photo Mikhail Kalashnikov, who invented the AK-47 assault rifle, attends festivities to celebrate his 90th birthday at the Kremlin in Moscow. (AP Photo/Natalia Kolesnikova, Pool, file) NEWLINE_CHAR NEWLINE_CHAR Mikhail Kalashnikov, designer of the legendary AK-47 assault rifle, turned to the head of the Russian Orthodox Church shortly before his death to express fears he was personally guilty for those it killed. NEWLINE_CHAR NEWLINE_CHAR Kalashnikov, who died in December at the age of 94, in April wrote a lengthy emotional letter to Russian Orthodox Church Patriarch Kirill, Izvestia, a pro-Kremlin daily, reported on Monday. NEWLINE_CHAR NEWLINE_CHAR \"My spiritual pain is unbearable. I keep having the same unsolved question: if my rifle took away people's lives, then can it be that I... am guilty for people's deaths, even if they were enemies?\" he asked. NEWLINE_CHAR NEWLINE_CHAR The typed letter on Kalashnikov's personal writing paper, reproduced by Izvestia, is signed with a wavering hand by the man who describes himself as \"a slave of God, the designer Mikhail Kalashnikov.\" NEWLINE_CHAR NEWLINE_CHAR Kalashnikov, whose funeral was attended by President Vladimir Putin, came up with the durable and simple rifle design after experiencing the Red Army's dire lack of weapons during World War II. NEWLINE_CHAR NEWLINE_CHAR Now the AK-47 is widely manufactured unlicenced around the world and has become a visual hallmark of armed insurgent movements, including those using child soldiers. NEWLINE_CHAR NEWLINE_CHAR Kalashnikov wrote that he first went into a church at the age of 91 and was later baptised. NEWLINE_CHAR NEWLINE_CHAR The Patriarch's press secretary, Alexander Volkov, told Izvestia that the Russian Church leader received the letter and wrote a personal reply. NEWLINE_CHAR NEWLINE_CHAR \"The Church has a very definite position: when weapons serve to protect the Fatherland, the Church supports both its creators and the soldiers who use it,\" Volkov said. NEWLINE_CHAR NEWLINE_CHAR \"He designed this rifle to defend his country, not so terrorists could use it in Saudi Arabia.\" NEWLINE_CHAR NEWLINE_CHAR The Russian Orthodox Church has sought to consolidate its new-found strength after the Soviet era by building up close ties with state agencies and powerful officials. NEWLINE_CHAR NEWLINE_CHAR When Kalashnikov was feted by the Soviet authorities, it would have been unthinkable for him to have declared himself anything else than an atheist. NEWLINE_CHAR NEWLINE_CHAR His daughter, Yelena, told Izvestia: \"Of course you can't say he went to services or lived strictly according to the commandments. You have to understand his generation.\"\n", 
    "Summary the following passages: Paragraph 1: Dragon's Lair II: Time Warp is a 1991 laserdisc video game by the Leland Corporation. It is the first true sequel to Dragon's Lair. As with the original, Dragon's Lair II: Time Warp consists of an animated short film that requires the player to move the joystick or press a fire button at certain times in order to continue. It takes place years after the original Dragon's Lair. Dirk has married Daphne, and the marriage has produced many children. When Daphne is kidnapped by the evil wizard Mordroc in order to be forced into marriage, Dirk's children and his mother-in-law are clearly upset by the abduction of Daphne, and Dirk must once again save her.\n\nParagraph 2: In 1914, he moved to Kraków and joined the First Cadre Company, which fought on the Austro-Hungarian side against Russia. In October 1914 he became a commander of a platoon of a squadron in . During the fighting in 1914–1915, he was promoted to lieutenant, and after the war he was awarded the V-Class Virtuti Militari. In August 1915, he moved to the special group in Warsaw. Soon he became an aide-de-camp to Józef Piłsudski. In 1918, he was sent on a mission to Russia. He was given three tasks: persuade General Józef Haller's army, then in the Ukraine, to back Piłsudski (he failed in this task); reach the French military mission in Moscow under General Lavergne (he succeeded in this task); and return from Moscow to Paris to liaise with the government there. Unfortunately, he was arrested by the Soviet Cheka as a member of the Polish Military Organisation while on a French diplomatic train on its way from Moscow to Murmansk (and Paris). He was imprisoned in the Taganka prison. He was freed thanks to the intervention of his future wife, Bronisława Wieniawa-Długoszowska, with the much-feared Cheka operative Yakovleva, then in charge of the prison. Bronisława, née Kliatchkin, was at that time married to the lawyer , the lawyer of Felix Dzerzhinsky, the head of the Cheka. She was a Lutheran, her family having converted from the Jewish faith when she was eight. He married her in a Lutheran ceremony on 2 October 1919 at Lutheran zbór in Nowy Gawłów. The marriage register records the details from her false French passport, including \"Lalande\" as her maiden name.\n\nParagraph 3: After the battle of Stamford Bridge and the death of Harald Hardrada, Skule and Ketel, the two sons of Tostig Godwinson, were taken to Norway under the wing of Hardrada's son Olaf.  Olaf, who became king of Norway, gave land to Ketel and arranged a good marriage for him; according to the 13th-century saga-writer Snorri Sturluson, \"from him are descended many great people\".  Skule became known as Skule Kongsfostre (king's foster-son), and was remembered as a remarkably intelligent and handsome man who commanded the king's hird.  He married a relative of the king, Gudrun Nevsteinsdotter, and their son was Åsolv of Rein (in Rissa), father of the lendmann Guttorm of Rein.  Guttorm's son, Bård Guttormsson of Rein, was a close friend and supporter of king Sverre Sigurdsson, fought alongside him in several battles, and was rewarded by being given the king's half-sister, Cecilia Sigurdsdotter, in marriage.  Bård and Cecilia's son Inge Bårdsson was born about the year 1185.  In 1204, when the child-king Guttorm Sigurdsson died, the two obvious candidates for the crown were Inge and his half-brother Haakon Galen, Cecilia's son by another husband.  After a struggle for power Inge was recognized as king, while Haakon retained his former command of the army.  This did not produce peace, for a faction known as the Bagler succeeded in splitting the kingdom, with Inge ruling the western half and their own candidate, Philip Simonsson, the eastern half.  Moreover, earl Haakon renewed his own claim to the crown, a claim which only lapsed with his death in 1214.  King Inge himself died in 1217.  Since 1213 the leader of the army and the hird had been Inge's half-brother Skule, a son of Bård Guttormsson by another wife, Ragnfrid Erlingsdotter, and therefore, like king Inge, fifth in descent in the male line from Tostig Godwinson.  The new king, Haakon IV, was no more than a boy, with Skule acting as his regent.  In 1225 Skule married his young daughter, Margrete, to the king, but this did not succeed in establishing perfect amity between the two men, and in 1239 Skule went into open rebellion, claiming the title of king for himself.  The subsequent war went against him, and he was killed in 1240.  From Haakon IV and Margrete Skulesdotter descend subsequent kings of Norway down to the present day.\n\nParagraph 4: The Israel Police is responsible for investigations and arrests regarding civilian crimes. If the Israel Police learns of a possible criminal offense through a complaint by a private citizen or through other evidence, it then decides whether or not to open an investigation. In the case of an offense other than a felony, a police officer with the rank of captain or higher is entitled to order that no investigation take place if the officer is of the opinion that no public interest is involved or another authority is legally competent to carry out the investigation. During a police investigation, a judge must issue a search warrant for police to search a home or review computer material, though a police officer may search a home without a warrant if there are reasonable grounds to assume a felony is being committed there or was recently committed there. Any search either with or without a warrant must be conducted in the presence of two witnesses who are not police officers unless the circumstances and urgency of the case do not allow it, a judge permitted it, or the owner of the property or one of the household members requested that it not be conducted in the presence of witnesses. If the police wish to arrest a suspect following an investigation, an arrest warrant must be obtained from a judge. The police must present evidence to the judge, who will issue a warrant only if satisfied that there is reasonable suspicion that the person committed an offense. A police officer is entitled to carry out an arrest without a warrant if there are reasonable grounds to suspect that the suspect committed an offense and if one of the following conditions are met: the suspected offense was committed in the officer's presence or in the recent past, there is a reasonable suspicion that the suspect will not appear for investigative procedures, there is reasonable suspicion that the suspect will disrupt trial proceedings, there is reasonable suspicion that the suspect's continued freedom will constitute a danger to the public, committed a select number of serious violent crimes, drug crimes, or security crimes, or there are reasonable grounds to suspect a suspect violated bail or escaped lawful custody.\n\n",
    "Passage 1:\nWaldrada of Lotharingia\nWaldrada was the mistress, and later the wife, of Lothair II of Lotharingia.\n\nBiography\nWaldrada's family origin is uncertain. The prolific 19th-century French writer Baron Ernouf suggested that Waldrada was of noble Gallo-Roman descent, sister of Thietgaud, the bishop of Trier, and niece of Gunther, archbishop of Cologne. However, these suggestions are not supported by any evidence, and more recent studies have instead suggested she was of relatively undistinguished social origins, though still from an aristocratic milieu.\nThe Vita Sancti Deicoli states that Waldrada was related to Eberhard II, Count of Nordgau (included Strasbourg) and the family of Etichonids, though this is a late 10th-century source and so may not be entirely reliable on this question.In 855 the Carolingian king Lothar II married Teutberga, a Carolingian aristocrat and the daughter of Bosonid Boso the Elder. The marriage was arranged by Lothar's father Lothar I for political reasons. It is very probable that Waldrada was already Lothar II's mistress at this time.Teutberga was allegedly not capable of bearing children and Lothar's reign was chiefly occupied by his efforts to obtain an annulment of their marriage, and his relations with his uncles Charles the Bald and Louis the German were influenced by his desire to obtain their support for this endeavour. Lothair, whose desire for annulment was arguably prompted by his affection for Waldrada, put away Teutberga. However, Hucbert took up arms on his sister's behalf, and after she had submitted successfully to the ordeal of water, Lothair was compelled to restore her in 858. Still pursuing his purpose, he won the support of his brother, Emperor Louis II, by a cession of lands and obtained the consent of the local clergy to the annulment and to his marriage with Waldrada, which took place in 862. However, Pope Nicholas I was suspicious of this and sent legates to investigate at the Council of Metz in 863. The Council found in favour of Lothair's divorce, which led to rumours that the papal legates may have bribed and thus meant that Nicholas order Lothair to take Teutberga back or face excommunication. \nWith the support of Charles the Bald and Louis the German, Teutberga appealed the annulment to Pope Nicholas. Nicholas refused to recognize the annulment and excommunicated Waldrada in 866, forcing Lothair to abandon Waldrada in favour of Teutberga. Lothair accepted this begrudgingly for a time, but shortly afterward at the end of 867 Pope Nicholas I died. Thus, Lothair began to seek the permission of the newly appointed Pope Adrian II to again put Teutberga aside and marry Waldrada, riding to Rome to speak with him on the matter in 869. However, on his way home, Lothair died.\n\nChildren\nWaldrada and Lothair II had some sons and probably three daughters, all of whom were declared illegitimate:\n\nHugh (c. 855–895), Duke of Alsace (867–885)\nGisela (c. 865–908), who in 883 married Godfrey, the Viking leader ruling in Frisia, who was murdered in 885\nBertha (c. 863–925), who married Theobald of Arles (c. 854–895), count of Arles, nephew of Teutberga. They had two sons, Hugh of Italy and Boso of Tuscany. After Theobald's death, between 895 and 898 she married Adalbert II of Tuscany (c. 875–915) They had at least three children: Guy, who succeeded his father as count and duke of Lucca and margrave of Tuscany, Lambert succeeded his brother in 929, but lost the titles in 931 to his half-brother Boso of Tuscany, and Ermengard.\nErmengarde (d. 90?)\nOdo (d. c.879)\nWhere was the wife of Francis I Rákóczi born?",
    "Passage 1:\nJim Ramel Kjellgren\nJim Love Ramel Kjellgren, (born 18 July 1987) is a Swedish actor. He is the son of Lotta Ramel and Johan H:son Kjellgren and the grandchild of Povel Ramel. He is perhaps best known as the character Jonte in the SVT series Eva & Adam, he reprised the role in the film Eva & Adam – fyra födelsedagar och ett fiasko.In 2020, Jim married Bernadette Gisele Hutson, who is French-American.\n\nFilmography\n1999–2000 – Eva & Adam (TV-series)\n2001 – Eva & Adam – fyra födelsedagar och ett fiasko\n2001 – Days Like This\n2004 – Kyrkogårdsön\n2005 – Storm\nPassage 2:\nTulasi (actress)\nTulasi (or Tulasi Shivamani) is an Indian actress who primarily works in Telugu, Kannada, and Tamil cinema. She started her career as a child actress. Later she appeared in lead actress and supporting actress roles. She has acted in over 300 films in Telugu, Kannada, Tamil, Malayalam, and Bhojpuri languages. She won two Nandi Awards and one Filmfare Award.\n\nCareer\nTulasi made her debut in the Telugu language when she was three months old in 1967. For a song in a film, a baby was needed and Tulasi was placed in the cradle after actress Savitri had requested Tulasi's mother, who was a friend of her. She was featured in a song when she was three-and-half years old in Jeevanatarangalu and said that she became a full-fledged actor when she was four. She had never been to school.She got married at age 28 to Kannada director Sivamani. She stated, \"I met him in the morning and by evening we tied the knot\". They have one son, Sai Tarun. Tulasi decided to quit acting after getting married, working only occasionally as a voice actor in Telugu films, including ones by Mani Ratnam. When her son was around six years old, she received several mother character roles. She initially declined them all, but finally signed on one Kannada film, Excuse Me, in which she played mother to Divya Spandana and which became a big hit. After that she was doing three films a year in Kannada.\nShe began to act mainly in mother roles in Telugu and Tamil film industries. Her notable supporting roles include performances in Sasirekha Parinayam, Mr. Perfect, Darling, Srimanthudu, Iddarammayilatho, Nenu Local, Mahanati & Dear Comrade in Telugu and Pillaiyar Theru Kadaisi Veedu, Easan, Mankatha, Sundarapandian, Aadhalal Kadhal Seiveer and Pandiya Naadu in Tamil. Tulasi has said that Aadhalal Kadhal Seiveer, in which she had played mother to Manisha Yadav's character, changed her life and brought her an \"identity as a screen mother\". Her portrayal of Chellamma in Pannaiyarum Padminiyum was praised too, with critics stating that she was \"brilliant\", and had given her \"career best performance\".\n\nPartial filmography\nAwards\nNandi AwardsBest Child Actress - Seetamalakshmi (1978)\nBest Child Actress - Sankarabharanam (1980)Filmfare Awards SouthFilmfare Award for Best Supporting Actress - Kannada - Josh\nPassage 3:\nStokkseyrar-Dísa\nThordis Markusdottir (Þórdís Markúsdóttir), known as Stokkseyrar-Dísa (1668–1728), was an Icelandic magician (Galdrmaster). She is known in history for her alleged magical powers. She is the subject of a least ten different folk sagas depicting her experiments within magic or Galdr.\nThordis Markusdottir belonged to the elite of the Iceland and was the grandchild of sheriff Torfi Erlendsson of Stafnes and related to Thormodus Torfæus, historian of the King of Denmark. She lived in Stokkseyri, thereby the name Stokkseyrar-Dísa. Some of the sagas around her centers on her magical duels with Eiríkur í Vogsósum.\nPassage 4:\nElizabeth (biblical figure)\nElizabeth (also spelled Elisabeth; Hebrew: אֱלִישֶׁבַע / אֱלִישָׁבַע \"My God has sworn\", Standard Hebrew: Elišévaʿ / Elišávaʿ, Tiberian Hebrew: ʾĔlîšéḇaʿ / ʾĔlîšāḇaʿ; Greek: Ἐλισάβετ Elisabet / Elisavet) was the mother of John the Baptist, the wife of Zechariah, and maternal aunt of Mary, mother of Jesus, according to the Gospel of Luke and in Islamic tradition. She was past normal child-bearing age when she conceived and gave birth to John.\n\nBiblical narrative\nAccording to the Gospel of Luke chapter 1, Elizabeth was \"of the daughters of Aaron\". She and her husband Zechariah/Zachariah were \"righteous before God, walking in all the commandments and ordinances of the Lord blameless\" (1:5–7), but childless. While he was in the temple of the Lord (1:8–12), Zachariah was visited by the angel Gabriel:\n\nBut the angel said to him: “Do not be afraid, Zechariah; your wife Elizabeth will bear you a son, and you are to call him John. He will be a joy and delight to you, and many will rejoice because of his birth, for he will be great in the sight of the Lord. He is never to take wine or other fermented drink, and he will be filled with the Holy Spirit even before he is born. Who is Sobe (Sister Of Saint Anne)'s grandchild?",
]

attention_diff_matric_kvblock = torch.zeros((NUM_LAYERS*NUM_HEADS, NUM_LAYERS*NUM_HEADS), device=DEVICE, dtype=torch.float32)

for prompt_i in range(len(prompt)):
  inputs = tokenizer(prompt[prompt_i], return_tensors="pt").to(DEVICE)
  print(f"Input Sequence Length: {inputs.input_ids.shape[1]}")

  outputs = model.generate(
    **inputs,
    max_length=16384,
    output_attentions=True,
    return_dict_in_generate=True
  )
  if method == "quest":
    model.quest_clear()

  generated_ids = outputs.sequences
  attentions = outputs.attentions # (output_tokens, num_layers, batch_size, num_heads, sequence_length, sequence_length)
  all_tokens = tokenizer.convert_ids_to_tokens(generated_ids[0])
  all_tokens = [token.replace("Ġ", "") for token in all_tokens]
  print(f"Generated Sequence Length: {len(all_tokens)}")
  print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])

  prompt_length = inputs.input_ids.shape[1]
  # Search for the most similar attention matric to each attention matric
  attentions = [list(step) for step in attentions]  # transfer tuple to list
  attentions = [[layer.to(DEVICE) for layer in step] for step in attentions]  # transfer to cuda
  num_layers = len(attentions[0])
  num_heads = attentions[0][0].shape[1]
  total_seq_len = len(all_tokens)
  block_num = ceil(total_seq_len / block_size)
  padded_seq_len = block_num * block_size
  current_attention_matric = torch.zeros((total_seq_len, total_seq_len), device=DEVICE)
  current_attention_matric_kvblock = torch.zeros((total_seq_len, block_num), device=DEVICE)
  compare_attention_matric = torch.zeros((num_heads, total_seq_len, total_seq_len), device=DEVICE)
  compare_attention_matric_kvblock = torch.zeros((num_heads, total_seq_len, block_num), device=DEVICE)
  padded_attention = torch.zeros(total_seq_len, padded_seq_len, device=DEVICE)
  mask = torch.zeros(total_seq_len, block_num, device=DEVICE)
  for i in range(total_seq_len):
      max_block_idx = min(i // block_size + 1, block_num)  # i 对应的最大 block 索引
      mask[i, :max_block_idx] = 1
  # output = "Layer\tHead\tSimilar Layer\tSimilar Head\tDiff"
  # open('results/similar_block_attention_result.txt', 'w', encoding='utf-8').write(output + '\n')
  for layer_i in range(1, num_layers):
      for head_i in range(num_heads):
          current_attention_matric.zero_()
          current_attention_matric[:attentions[0][layer_i][0, head_i].shape[0], :attentions[0][layer_i][0, head_i].shape[0]] = attentions[0][layer_i][0, head_i].detach()
          for i in range(1, len(attentions)):
              current_seq_len = prompt_length + i
              current_attention_matric[prompt_length + i - 1, :current_seq_len] = attentions[i][layer_i][0, head_i].detach()
          current_attention_matric_kvblock.zero_()
          padded_attention.zero_()
          padded_attention[:, :total_seq_len] = current_attention_matric
          reshaped_attention = padded_attention.view(total_seq_len, block_num, block_size)
          current_attention_matric_kvblock = torch.sum(reshaped_attention, dim=-1)
          _, indices = torch.topk(current_attention_matric_kvblock, k=topk, dim=1)
          # print("indices: ", indices)
          current_attention_matric_kvblock.zero_()
          current_attention_matric_kvblock.scatter_(1, indices, 1)
          # Change current_attention_matric_kvblock to the lower triangular matrix
          current_attention_matric_kvblock *= mask
          current_attention_matric_kvblock[:prompt_length] = mask[:prompt_length] # 将prefill阶段的attention置为下三角
          # print("current_attention_matric_kvblock: ", current_attention_matric_kvblock)
          # plt.figure(figsize=(10, 8))
          # sns.heatmap(current_attention_matric_kvblock.cpu().numpy(), cmap="viridis")
          # plt.xlabel("Block Key Tokens")
          # plt.ylabel("Query Tokens")
          # plt.title(f"Block KV Attention Heatmap")
          # plt.show()
          # # save the attention matric figure
          # plt.savefig(f"block_attention_heatmap_layer_{layer_i}_head_{head_i}.png")
          # plt.close()
          # similar_layer_head = (layer_i, head_i)
          # min_diff = float("inf")
          for layer_j in range(layer_i):
              for head_j in range(len(attentions[0][layer_j][0])):
                  compare_attention_matric[head_j, :attentions[0][layer_j][0, head_j].shape[0], :attentions[0][layer_j][0, head_j].shape[0]] = attentions[0][layer_j][0, head_j].detach()
                  for i in range(1, len(attentions)):
                      current_seq_len = prompt_length + i
                      compare_attention_matric[head_j, prompt_length + i - 1, :current_seq_len] = attentions[i][layer_j][0, head_j].detach()
                  compare_attention_matric_kvblock[head_j].zero_()
                  padded_attention.zero_()
                  padded_attention[:, :total_seq_len] = compare_attention_matric[head_j]
                  reshaped_attention = padded_attention.view(total_seq_len, block_num, block_size)
                  compare_attention_matric_kvblock[head_j] = torch.sum(reshaped_attention, dim=-1)
                  _, indices = torch.topk(compare_attention_matric_kvblock[head_j], k=topk, dim=1)
                  compare_attention_matric_kvblock[head_j].zero_()
                  compare_attention_matric_kvblock[head_j].scatter_(1, indices, 1)
                  # Change compare_attention_matric_kvblock to the lower triangular matrix
                  compare_attention_matric_kvblock[head_j] *= mask
                  compare_attention_matric_kvblock[head_j][:prompt_length] = mask[:prompt_length] # 将prefill阶段的attention置为下三角
                  # plt.figure(figsize=(10, 8))
                  # sns.heatmap(compare_attention_matric_kvblock[head_j].cpu().numpy(), cmap="viridis")
                  # plt.xlabel("Block Key Tokens")
                  # plt.ylabel("Query Tokens")
                  # plt.title(f"Block KV Attention Heatmap")
                  # plt.show()
                  # # save the attention matric figure
                  # plt.savefig(f"block_attention_heatmap_layer_{layer_j}_head_{head_j}.png")
                  # plt.close()

              current_expanded = current_attention_matric_kvblock.unsqueeze(0).expand(num_heads, -1, -1)
              diffs = torch.sum(torch.abs(current_attention_matric_kvblock - compare_attention_matric_kvblock), dim=(1, 2)) # [num_heads]
              attention_diff_matric_kvblock[layer_i * num_heads + head_i, layer_j * num_heads:(layer_j + 1) * num_heads] += diffs
              with open("results/attention_diff_kvblock.jsons", "a") as f:
                json.dump({"layer_i": layer_i, "head_i": head_i, "head_j": head_j, "diffs": diffs.cpu().tolist()}, f)
                f.write("\n")
              # diff, min_index = torch.min(diffs, dim=0)
              # if diff < min_diff:
              #     min_diff = diff
              #     similar_layer_head = (layer_j, min_index)
          # print(f"Layer {layer_i}, Head {head_i} is most similar to Layer {similar_layer_head[0]}, Head {similar_layer_head[1]} with diff {min_diff}")
          # output = f"{layer_i}\t{head_i}\t{similar_layer_head[0]}\t{similar_layer_head[1]}\t{min_diff}"
          # open('results/similar_block_attention_result.txt', 'a', encoding='utf-8').write(output + '\n')
          with open('results/attention_diff_matric_kvblock.json', 'w') as f:
            json.dump(attention_diff_matric_kvblock.cpu().tolist(), f)

attention_diff_matric_kvblock /= len(prompt)
with open('results/attention_diff_matric_kvblock.json', 'w') as f:
  json.dump(attention_diff_matric_kvblock.cpu().tolist(), f)
print("Attention diff matric: ", attention_diff_matric_kvblock.cpu().tolist())
similar_layer_head = torch.zeros((NUM_LAYERS * NUM_HEADS,), device=DEVICE)
for i in range(NUM_LAYERS * NUM_HEADS):
  similar_layer_head[i] = torch.argmin(attention_diff_matric_kvblock[i])
with open('results/similar_layer_head.json', 'w') as f:
  json.dump(similar_layer_head.cpu().tolist(), f)
print("Similar layer head: ", similar_layer_head.cpu().tolist())
