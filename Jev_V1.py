#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Jev bira jedan broj, zatim sledeći uz već odabrane brojeve.

    - Loto 7/39.
    - Tri nezavisna CSV-a.
    - Cela istorija svakog CSV-a.
    - Jedna završna NEXT kombinacija po CSV-u.
    - Bez lokalnog generisanja brojeva.
    - Bez promene Jev izbora i bez kozmetičkih filtera.

CSV format:
    Bez zaglavlja, sedam različitih celih brojeva 1..39 po redu.
    Hronološki: najstarije izvlačenje prvo, najnovije poslednje.

Podrazumevani API ugovor preuzet iz:
    POST https://api.typesafe.ai/
    Authorization: 
    JSON: {"model": ..., "state": ..., "questions": ...}

API ključ unosi se lokalno pri pokretanju, ili preko promenljive TYPESAFE_API_KEY.

Program:
    - čita samo navedene CSV fajlove;
    - šalje istoriju Jev servisu radi izbora;
    - rezultate ispisuje u terminal;

Ne skraćuje istoriju ako servis odbije veličinu zahteva.
Ne ponavlja automatski API pozive posle greške.
"""


import argparse
import csv
import getpass
import hashlib
import json
import math
import os
import sys
import urllib.error
import urllib.request

from dataclasses import dataclass
from pathlib import Path


API_URL = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"

DEFAULT_FILES = (
    "/data/loto7_4704_k83.csv",
    "/data/loto7_4704_k83_loto_2974.csv",
    "/data/loto7_4704_k83_loto_plus_1730.csv",
)

MAX_RESPONSE_BYTES = 8 * 1024 * 1024

INSTRUCTION = (
    "Choose exactly one legal number for the next Loto 7/39 draw. "
    "Use the complete supplied draw history and current_selection. "
    "History rows are chronological, oldest first. "
    "Each historical row is an unordered seven-number combination; "
    "its displayed numeric order is not the extraction order. "
    "Condition this choice on numbers already selected for this "
    "same next combination. "
    "Choose only from the supplied criteria. "
    "The final combination must contain seven distinct integers "
    "from 1 through 39. "
    "Do not impose artificial rules about consecutive numbers, "
    "lower or upper halves, parity, spacing, or novelty. "
    "Return the normal choice answer with candidate probabilities "
    "and confidence required by the API."
)


class JevError(RuntimeError):
    pass


@dataclass(frozen=True)
class Dataset:
    path: Path
    draws: tuple
    fingerprint: str


@dataclass(frozen=True)
class ChoiceStep:
    position: int
    number: int
    probability: float
    confidence: float
    probability_sum: float
    model: str
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class Prediction:
    dataset: Dataset
    combination: tuple
    steps: tuple

    @property
    def selection_score(self):
        probabilities = [step.probability for step in self.steps]

        if any(value == 0.0 for value in probabilities):
            return 0.0

        return math.exp(
            sum(math.log(value) for value in probabilities)
            / len(probabilities)
        )


def read_dataset(filename):
    path = Path(filename).expanduser().resolve(strict=True)

    if not path.is_file():
        raise ValueError(f"Nije običan fajl: {path}")

    draws = []

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)

        for line_number, row in enumerate(reader, 1):
            if not row or all(not cell.strip() for cell in row):
                continue

            if len(row) != 7:
                raise ValueError(
                    f"{path.name}, red {line_number}: "
                    f"očekivano 7 kolona; dobijeno {len(row)}."
                )

            try:
                values = tuple(int(cell.strip()) for cell in row)
            except ValueError as exc:
                raise ValueError(
                    f"{path.name}, red {line_number}: "
                    "dozvoljeni su samo celi brojevi."
                ) from exc

            if any(number < 1 or number > 39 for number in values):
                raise ValueError(
                    f"{path.name}, red {line_number}: "
                    "svi brojevi moraju biti u opsegu 1..39."
                )

            if len(set(values)) != 7:
                raise ValueError(
                    f"{path.name}, red {line_number}: "
                    "kombinacija ima ponovljen broj."
                )

            # Sortira se samo kombinacija, nikada redovi istorije.
            draws.append(tuple(sorted(values)))

    if not draws:
        raise ValueError(f"Prazan CSV: {path}")

    canonical = "\n".join(
        ",".join(map(str, draw)) for draw in draws
    ).encode("ascii")

    fingerprint = hashlib.sha256(canonical).hexdigest()

    return Dataset(
        path=path,
        draws=tuple(draws),
        fingerprint=fingerprint,
    )


def probability(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JevError(f"{label}: očekivana numerička vrednost.")

    result = float(value)

    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise JevError(f"{label}: vrednost mora biti između 0 i 1.")

    return result


def token_count(usage, name):
    value = usage.get(name, 0)

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise JevError(f"Neispravan usage.{name}.")

    return value


def reject_json_constant(value):
    raise ValueError(f"Nedozvoljena JSON konstanta: {value}")


class JevClient:
    def __init__(self, api_key, model, timeout):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def decide(self, state, question):
        payload = {
            "model": self.model,
            "state": state,
            "questions": {"pick": question},
        }

        body = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")

        request = urllib.request.Request(
            API_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout
            ) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)

        except urllib.error.HTTPError as exc:
            detail = exc.read(3000).decode("utf-8", errors="replace")
            detail = detail.replace(self.api_key, "[SKRIVENO]")

            extra = ""
            if exc.code in (400, 413, 422):
                extra = (
                    " Poslata je cela istorija; program je neće "
                    "automatski skratiti."
                )

            raise JevError(
                f"Jev HTTP {exc.code}.{extra}\n"
                f"Odgovor servisa: {detail[:1500]}"
            ) from exc

        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise JevError(
                "Jev mrežni zahtev nije uspešno završen. "
                "Nema automatskog ponavljanja; servis je možda "
                "već obradio zahtev."
            ) from exc

        if len(raw) > MAX_RESPONSE_BYTES:
            raise JevError("Odgovor servisa prelazi dozvoljenu veličinu.")

        try:
            data = json.loads(
                raw.decode("utf-8"),
                parse_constant=reject_json_constant,
            )
        except (UnicodeError, ValueError) as exc:
            raise JevError("Servis nije vratio ispravan JSON.") from exc

        return self.validate_response(data, question)

    @staticmethod
    def validate_response(data, question):
        if not isinstance(data, dict):
            raise JevError("API odgovor nije JSON objekat.")

        answers = data.get("answers")
        usage = data.get("usage")
        model = data.get("model")

        if not isinstance(answers, dict) or set(answers) != {"pick"}:
            raise JevError("API odgovor nema tačno traženo pitanje 'pick'.")

        if not isinstance(usage, dict):
            raise JevError("API odgovor nema ispravan objekat 'usage'.")

        if not isinstance(model, str) or not model.strip():
            raise JevError("API odgovor nema naziv stvarno korišćenog modela.")

        answer = answers["pick"]

        if not isinstance(answer, dict):
            raise JevError("Odgovor na 'pick' nije objekat.")

        if answer.get("type") != "choice":
            raise JevError("Odgovor nije tipa 'choice'.")

        candidates = set(question["criteria"])
        choice = answer.get("choice")

        if not isinstance(choice, str) or choice not in candidates:
            raise JevError("Jev je vratio izbor van ponuđenih kandidata.")

        probabilities = answer.get("probabilities")

        if (
            not isinstance(probabilities, dict)
            or set(probabilities) != candidates
        ):
            raise JevError(
                "Jev nije vratio raspodelu za tačno ponuđene kandidate."
            )

        checked = {
            candidate: probability(value, f"probabilities[{candidate}]")
            for candidate, value in probabilities.items()
        }

        total = math.fsum(checked.values())

        # Tolerancija za zaokruživanje izlaza servisa.
        # Vrednosti se ne normalizuju i ne prepravljaju.
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=0.02):
            raise JevError(
                "Zbir kandidatskih verovatnoća nije približno 1: "
                f"{total:.8f}. Odgovor nije prihvaćen."
            )

        confidence = probability(
            answer.get("confidence"), "confidence"
        )

        return {
            "number": int(choice),
            "probability": checked[choice],
            "probability_sum": total,
            "confidence": confidence,
            "model": model,
            "input_tokens": token_count(usage, "input_tokens"),
            "output_tokens": token_count(usage, "output_tokens"),
        }


def predict_dataset(client, dataset):
    chosen = []
    steps = []

    # Kompaktan prikaz čuva svako izvlačenje i njegov redosled.
    # Nema datuma, izmišljenih brojeva kola ni prozora od 30 dana.
    history = [list(draw) for draw in dataset.draws]

    for position in range(1, 8):
        candidates = [
            number for number in range(1, 40)
            if number not in chosen
        ]

        state = {
            "game": "Loto 7/39",
            "rules": {
                "main": {
                    "minimum": 1,
                    "maximum": 39,
                    "count": 7,
                    "unique": True,
                }
            },
            "history_order": "oldest_to_newest",
            "history_rows": len(history),
            "history": history,
            "current_selection": {"main": list(chosen)},
            "target": "the draw immediately after the last history row",
        }

        question = {
            "type": "choice",
            "instructions": (
                INSTRUCTION
                + f" Select number {position} of 7."
            ),
            "criteria": {
                f"{number:02d}": None for number in candidates
            },
        }

        print(
            f"  Jev izbor {position}/7; "
            f"cela istorija: {len(history)} redova...",
            flush=True,
        )

        result = client.decide(state, question)
        number = result["number"]

        if number in chosen or not 1 <= number <= 39:
            raise JevError("Neispravan izbor; NEXT nije kompletiran.")

        chosen.append(number)

        step = ChoiceStep(
            position=position,
            number=number,
            probability=result["probability"],
            confidence=result["confidence"],
            probability_sum=result["probability_sum"],
            model=result["model"],
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
        )
        steps.append(step)

        print(
            f"    Izabrano: {number:02d} | "
            f"Jev preferencija: {step.probability:.6f} | "
            f"confidence: {step.confidence:.6f}",
            flush=True,
        )

    combination = tuple(sorted(chosen))

    if len(combination) != 7 or len(set(combination)) != 7:
        raise JevError("Završna kombinacija nije validna za 7/39.")

    return Prediction(
        dataset=dataset,
        combination=combination,
        steps=tuple(steps),
    )


def print_prediction(prediction):
    print("\n" + "=" * 72)
    print(prediction.dataset.path.name)
    print(f"Istorijskih izvlačenja: {len(prediction.dataset.draws)}")
    print(f"SHA-256 sadržaja: {prediction.dataset.fingerprint}")
    print(
        "Redosled Jev izbora: "
        + ", ".join(str(step.number) for step in prediction.steps)
    )
    print(
        "NEXT: "
        + ", ".join(map(str, prediction.combination))
    )
    print(
        "Jev selection score: "
        f"{prediction.selection_score:.8f}"
    )
    print(
        "Prosečni Jev confidence: "
        f"{sum(step.confidence for step in prediction.steps) / 7:.8f}"
    )

    models = sorted({step.model for step in prediction.steps})
    print("Stvarno korišćeni modeli: " + ", ".join(models))
    print(
        "Tokeni koje je prijavio servis: "
        f"ulaz={sum(step.input_tokens for step in prediction.steps)}, "
        f"izlaz={sum(step.output_tokens for step in prediction.steps)}"
    )
    print("Selection score i confidence su izlazi Jev servisa.")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Izvorni Jev sekvencijalni izbor, "
            "prilagođen za tri CSV-a i Loto 7/39."
        )
    )

    parser.add_argument(
        "csv",
        nargs="*",
        help="Tri CSV putanje; bez argumenata koristi DEFAULT_FILES.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("TYPESAFE_MODEL", DEFAULT_MODEL),
        help="Jev model; podrazumevano TYPESAFE_MODEL ili jev-latest.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Vremensko ograničenje jednog API poziva u sekundama.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    filenames = args.csv or DEFAULT_FILES

    if len(filenames) != 3:
        raise ValueError("Potrebne su tačno tri CSV putanje.")

    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("--timeout mora biti pozitivan konačan broj.")

    model = args.model.strip()

    if not model:
        raise ValueError("Naziv Jev modela ne sme biti prazan.")

    # Svi ulazi proveravaju se pre prvog API poziva.
    datasets = [read_dataset(filename) for filename in filenames]

    if len({dataset.path for dataset in datasets}) != 3:
        raise ValueError("Navedi tri različita CSV fajla.")

    print("JEV LOTO V1")
    print(f"Traženi model: {model}")
    print("Tri odvojene istorije; sedam sekvencijalnih izbora po istoriji.")
    print("Ukupno 21 API poziv ako se svi zahtevi uspešno završe.")
    print("Svaki poziv šalje celu istoriju odgovarajućeg CSV-a.")

    for dataset in datasets:
        print(
            f"  {dataset.path.name}: "
            f"{len(dataset.draws)} izvlačenja"
        )

    if len({dataset.fingerprint for dataset in datasets}) != 3:
        print(
            "NAPOMENA: neki CSV fajlovi imaju identičan "
            "normalizovan sadržaj."
        )

    api_key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    # api_key = "feofczgk...49YRxHXTndQYfYet1"

    if not api_key:
        if not sys.stdin.isatty():
            raise ValueError(
                "Postavi TYPESAFE_API_KEY pre neinteraktivnog pokretanja."
            )

        api_key = getpass.getpass(
            "\nJev TYPESAFE_API_KEY — unos se ne prikazuje: "
        ).strip()

    if not api_key:
        raise ValueError("API ključ nije unet.")

    client = JevClient(api_key, model, args.timeout)
    predictions = []

    for dataset in datasets:
        print(f"\nObrada: {dataset.path.name}", flush=True)
        prediction = predict_dataset(client, dataset)
        predictions.append(prediction)
        print_prediction(prediction)

    print("\n" + "=" * 72)
    print("ZAVRŠNI NEXT PREGLED")

    for prediction in predictions:
        print(
            f"{prediction.dataset.path.name}: "
            + ", ".join(map(str, prediction.combination))
        )

    identical = {}

    for prediction in predictions:
        identical.setdefault(
            prediction.combination, []
        ).append(prediction.dataset.path.name)

    for combination, names in identical.items():
        if len(names) > 1:
            print(
                "\nIsti NEXT: "
                + " / ".join(names)
                + ": "
                + ", ".join(map(str, combination))
            )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "\nPrekinuto. Nedovršena kombinacija nije završni NEXT.",
            file=sys.stderr,
        )
        sys.exit(130)
    except (JevError, OSError, ValueError, csv.Error) as exc:
        print(f"\nGREŠKA: {exc}", file=sys.stderr)
        print(
            "Nema zamenskih kombinacija ni automatskog "
            "ponavljanja neuspelog zahteva.",
            file=sys.stderr,
        )
        sys.exit(1)
