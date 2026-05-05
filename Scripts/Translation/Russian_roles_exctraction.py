import pandas as pd

INPUT = r"C:\Users\olemga\OneDrive - PRIO\Documents\MA Thesis\Methods\Data\Input\lenta_news_1999_2024.csv"
OUTPUT = r"C:\Users\olemga\OneDrive - PRIO\Documents\MA Thesis\Methods\Data\Input\lenta_news_1999_2024_subset.csv"

CHUNKSIZE = 20000

ROLE_CANON_RU = {
    # professors / teaching staff
    "профессор": "professor",
    "профессор университета": "professor",
    "профессор кафедры": "professor",
    "профессор права": "professor",
    "профессор политологии": "political scientist",
    "профессор экономики": "economist",
    "профессор социологии": "sociologist",
    "профессор истории": "historian",
    "доцент": "professor",
    "старший преподаватель": "lecturer",
    "преподаватель": "lecturer",
    "лектор": "lecturer",

    # researchers / fellows / scholars
    "исследователь": "researcher",
    "научный сотрудник": "researcher",
    "старший научный сотрудник": "researcher",
    "ведущий научный сотрудник": "researcher",
    "младший научный сотрудник": "researcher",
    "приглашённый исследователь": "researcher",
    "приглашенный исследователь": "researcher",
    "постдок": "researcher",
    "постдокторант": "researcher",
    "научный сотрудник института": "researcher",
    "научный сотрудник центра": "researcher",
    "научный сотрудник университета": "researcher",

    "феллоу": "fellow",
    "научный феллоу": "fellow",
    "старший феллоу": "fellow",
    "приглашённый феллоу": "fellow",
    "приглашенный феллоу": "fellow",

    "учёный": "scholar",
    "ученый": "scholar",
    "ведущий учёный": "scholar",
    "ведущий ученый": "scholar",

    # generic scientist + disciplines
    "учёный-экономист": "economist",
    "ученый-экономист": "economist",
    "учёный-политолог": "political scientist",
    "ученый-политолог": "political scientist",

    "учёный-физик": "physics",
    "ученый-физик": "physics",
    "учёный-химик": "chemistry",
    "ученый-химик": "chemistry",

    "учёный-историк": "historian",
    "ученый-историк": "historian",

    "учёный-юрист": "law",
    "ученый-юрист": "law",

    "учёный-социолог": "sociologist",
    "ученый-социолог": "sociologist",

    "учёный-психолог": "psychologist",
    "ученый-психолог": "psychologist",

    "учёный-биолог": "biology",
    "ученый-биолог": "biology",

    "учёный-эпидемиолог": "epidemiology",
    "ученый-эпидемиолог": "epidemiology",

    "учёный-вирусолог": "virology",
    "ученый-вирусолог": "virology",

    "учёный-климатолог": "climatology",
    "ученый-климатолог": "climatology",

    "учёный-демограф": "demography",
    "ученый-демограф": "demography",

    # single-word discipline roles
    "политолог": "political scientist",
    "экономист": "economist",
    "социолог": "sociologist",
    "антрополог": "anthropologist",
    "психолог": "psychologist",
    "историк": "historian",
    "демограф": "demography",
    "статистик": "statistics",
    "эпидемиолог": "epidemiology",
    "вирусолог": "virology",
    "биолог": "biology",
    "физик": "physics",
    "химик": "chemistry",
    "географ": "geography",
    "климатолог": "climatology",

    # analysts
    "аналитик": "analyst",
    "политический аналитик": "analyst",
    "военный аналитик": "analyst",
    "аналитик по безопасности": "analyst",
    "эксперт-аналитик": "analyst",
    "финансовый аналитик": "analyst",
    "рыночный аналитик": "analyst",
    "аналитик энергетики": "analyst",
    "аналитик по энергетике": "analyst",

    # directors / program people
    "директор по исследованиям": "research director",
    "директор исследований": "research director",
    "руководитель исследовательской программы": "program director",
    "руководитель программы": "program director",
    "директор центра": "program director",
    "руководитель центра": "program director",

    # generic experts / specialists
    "эксперт": "expert",
    "ведущий эксперт": "expert",
    "независимый эксперт": "expert",
    "эксперт по безопасности": "expert",
    "эксперт по международным отношениям": "foreign policy",
    "эксперт по внешней политике": "foreign policy",
    "эксперт по России": "russia studies",
    "эксперт по российской политике": "russia studies",
    "эксперт по постсоветскому пространству": "russia studies",
    "кремлиновед": "russia studies",

    "специалист": "specialist",
    "ведущий специалист": "specialist",
    "специалист по международным отношениям": "foreign policy",
    "специалист по России": "russia studies",
    "специалист по энергетике": "energy analyst",
    "специалист по кибербезопасности": "expert",
    "специалист по праву": "law",

    # law / legal scholar
    "правовед": "law",
    "юрист": "law",
    "конституционалист": "law",

    # public health / medical research
    "медицинский исследователь": "researcher",
    "эксперт в области общественного здоровья": "public health",
    "специалист по общественному здоровью": "public health",
    "эксперт по здравоохранению": "public health",
}

ROLE_SURFACES_RU = sorted(set(ROLE_CANON_RU.keys()))


def detect_roles_ru(text: str) -> str:
    """
    Return a semicolon-separated string of canonical roles
    found in the Russian text. Empty string if none.
    """
    if not isinstance(text, str):
        return ""
    t = text.lower()
    found = set()
    for surface, canon in ROLE_CANON_RU.items():
        if surface in t:
            found.add(canon)
    if not found:
        return ""
    return ";".join(sorted(found))


def main():
    first = True
    total_rows = 0
    kept_rows = 0


    usecols = ["date", "title", "article", "period"]

    for i, chunk in enumerate(pd.read_csv(INPUT, chunksize=CHUNKSIZE, usecols=usecols)):
        print(f"Processing chunk {i}...")

        # Rename article -> text so the rest of the script works unchanged
        chunk = chunk.rename(columns={"article": "text"})

        # compute key_role from Russian text
        chunk["key_role"] = chunk["text"].apply(detect_roles_ru)

        # keep only rows where we found at least one role
        expert_chunk = chunk[chunk["key_role"] != ""]
        total_rows += len(chunk)
        kept_rows += len(expert_chunk)

        if not expert_chunk.empty:
            mode = "w" if first else "a"
            header = first
            expert_chunk.to_csv(
                OUTPUT,
                mode=mode,
                header=header,
                index=False,
                encoding="utf-8"
            )
            first = False

    print(f"Done. Scanned {total_rows} rows, kept {kept_rows} expert-rows.")
    print("Output file:", OUTPUT)


if __name__ == "__main__":
    main()
