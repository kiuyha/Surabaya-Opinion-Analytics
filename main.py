from src.scrapping import scrap_nitter

arr = scrap_nitter("Surabaya (keluhan OR komplain OR lapor OR aduan OR masalah OR parah OR jelek OR buruk OR mengecewakan OR layanan OR penanganan) (min OR tolong OR @) lang:id -filter:retweets", time_budget=1*60*60)