from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data yg dilatih dan di test"

BASE_DATASET = DATA_DIR / "(1)dataset_intent_final_ready_training.csv"
NOISE_ADDON = DATA_DIR / "dataset_intent_noise_ambiguous_addon.csv"
LEGACY_HARD_TEST = DATA_DIR / "dataset_intent_hard_manual_test.csv"

TRAIN_OUTPUT = DATA_DIR / "dataset_intent_13_ready_training.csv"
HARD_TEST_OUTPUT = DATA_DIR / "dataset_intent_13_hard_test.csv"
REPORT_OUTPUT = DATA_DIR / "intent_dataset_report.json"
CONTRACT_PATH = ROOT / "intent_contract.json"

MIN_TRAIN_ROWS_PER_LABEL = 100


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def load_contract_labels() -> tuple[str, ...]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    return tuple(item["name"] for item in contract["labels"])


CANONICAL_LABELS = load_contract_labels()
CANONICAL_LABEL_SET = set(CANONICAL_LABELS)


def read_repaired_csv(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header or [part.strip() for part in header] != ["text", "label"]:
            raise ValueError(f"{path.name}: header harus text,label")

        for line_number, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) < 2:
                raise ValueError(f"{path.name}:{line_number}: kolom tidak lengkap")

            # Dataset lama memiliki beberapa koma yang tidak diberi quote.
            # Kolom terakhir selalu label; bagian sebelumnya adalah teks.
            text = ",".join(row[:-1]).strip()
            label = row[-1].strip()

            if not text or not label:
                raise ValueError(f"{path.name}:{line_number}: teks/label kosong")

            rows.append({"text": text, "label": label})

    return rows


def add_unique_rows(
    target: list[dict[str, str]],
    seen: dict[str, str],
    rows: list[dict[str, str]],
) -> int:
    added = 0
    for row in rows:
        text = re.sub(r"\s+", " ", str(row["text"]).strip())
        label = str(row["label"]).strip()
        normalized = normalize_text(text)

        if label not in CANONICAL_LABEL_SET:
            raise ValueError(f"Label tidak dikenal: {label!r} pada {text!r}")

        previous_label = seen.get(normalized)
        if previous_label and previous_label != label:
            raise ValueError(
                f"Konflik label untuk {text!r}: {previous_label} vs {label}"
            )
        if previous_label:
            continue

        seen[normalized] = label
        target.append({"text": text, "label": label})
        added += 1

    return added


def generated_greetings(limit: int = 110) -> list[dict[str, str]]:
    openers = [
        "halo",
        "hallo",
        "hai",
        "hi",
        "hello",
        "selamat pagi",
        "pagi",
        "selamat siang",
        "siang",
        "selamat sore",
        "sore",
        "selamat malam",
        "malam",
        "permisi",
        "misi",
        "assalamualaikum",
        "punten",
        "halo robot jadul",
        "hai robot jadul",
        "hello robot jadul",
    ]
    endings = [
        "",
        "kak",
        "min",
        "admin",
        "gan",
        "boleh tanya",
        "mau tanya dulu",
        "bisa bantu",
        "saya baru mampir",
        "aku baru lihat tokonya",
    ]

    rows = []
    for ending in endings:
        for opener in openers:
            text = f"{opener} {ending}".strip()
            rows.append({"text": text, "label": "greeting"})
            if len(rows) >= limit:
                return rows
    return rows


def generated_compare_samples(limit: int = 150) -> list[dict[str, str]]:
    pairs = [
        ("voltes v", "grendizer"),
        ("mazinger z", "getter robo"),
        ("chogokin", "figure biasa"),
        ("voltron", "megazord"),
        ("daimos", "combattler v"),
        ("gaiking", "dancouga"),
        ("goldrake", "mazinger z"),
        ("gundam", "super robot"),
        ("getter robo", "voltes v"),
        ("grendizer", "gaiking"),
        ("figure robot", "model kit"),
        ("produk bandai", "produk takara"),
        ("robot vintage", "robot modern"),
        ("chogokin gx", "chogokin deluxe"),
        ("voltron vintage", "voltron modern"),
    ]
    templates = [
        "bandingkan {a} dengan {b}",
        "tolong bandingkan {a} dan {b}",
        "{a} vs {b}",
        "{a} versus {b}",
        "apa bedanya {a} dengan {b}",
        "perbedaan {a} dan {b} apa",
        "lebih bagus {a} atau {b}",
        "bagusan {a} atau {b}",
        "mending pilih {a} atau {b}",
        "mana yang lebih worth it antara {a} dan {b}",
        "untuk koleksi lebih cocok {a} atau {b}",
        "untuk pajangan pilih {a} atau {b}",
        "dari sisi harga lebih baik {a} atau {b}",
        "dari sisi kondisi mending {a} atau {b}",
        "kalau harus pilih satu ambil {a} atau {b}",
    ]

    rows = []
    for template in templates:
        for first, second in pairs:
            rows.append(
                {
                    "text": template.format(a=first, b=second),
                    "label": "compare",
                }
            )
            if len(rows) >= limit:
                return rows
    return rows


def generated_tracking_samples(limit: int = 150) -> list[dict[str, str]]:
    shipments = [
        ("jne", "JNE1234567890"),
        ("j&t", "JP1234567890"),
        ("sicepat", "SC1234567890"),
        ("anteraja", "AJ1234567890"),
        ("pos", "POS123456789"),
        ("tiki", "TK1234567890"),
        ("ninja", "NV1234567890"),
        ("lion parcel", "LP1234567890"),
        ("sap express", "SAP123456789"),
        ("id express", "IDE123456789"),
    ]
    templates = [
        "cek resi {number} kurir {courier}",
        "tolong lacak resi {courier} {number}",
        "paket dengan resi {number} sudah sampai mana",
        "tracking paket {courier} nomor {number}",
        "posisi paket saya dengan resi {number} di mana",
        "bisa cek perjalanan paket {number}",
        "resi {number} masih dalam perjalanan atau tidak",
        "paket {courier} {number} kapan sampai",
        "status pengiriman resi {number}",
        "lacak paket saya pakai {courier} nomor {number}",
        "kenapa resi {number} belum bergerak",
        "paket dengan nomor {number} sudah diterima belum",
        "cek update terakhir paket {courier} {number}",
        "barang saya di kurir {courier} sudah sampai mana",
        "tolong cek lokasi paket untuk resi {number}",
    ]

    rows = []
    for template in templates:
        for courier, number in shipments:
            rows.append(
                {
                    "text": template.format(courier=courier, number=number),
                    "label": "shipment_tracking",
                }
            )
            if len(rows) >= limit:
                return rows
    return rows


def generated_out_of_scope_general(limit: int = 70) -> list[dict[str, str]]:
    questions = [
        "jam berapa sekarang",
        "sekarang hari apa",
        "tanggal berapa hari ini",
        "bagaimana cuaca hari ini",
        "besok akan hujan atau tidak",
        "berapa suhu udara sekarang",
        "siapa presiden indonesia",
        "siapa gubernur jakarta",
        "apa ibu kota jepang",
        "ceritakan sejarah indonesia",
        "siapa penemu telepon",
        "planet terbesar apa",
        "berapa hasil dua tambah dua",
        "hitung akar kuadrat seratus",
        "tolong kerjakan soal matematika",
        "buatkan kode python",
        "cara membuat website",
        "jelaskan javascript",
        "buatkan query sql",
        "cara memperbaiki komputer",
        "resep nasi goreng",
        "cara membuat kue",
        "menu makan malam apa",
        "terjemahkan kalimat ini",
        "buatkan puisi",
        "ceritakan dongeng",
        "berita terbaru hari ini",
        "berapa skor pertandingan tadi",
        "jadwal pertandingan bola kapan",
        "apa ramalan zodiak hari ini",
        "film terbaru yang bagus apa",
        "lagu yang sedang populer apa",
        "cara belajar bahasa inggris",
        "jelaskan fotosintesis",
        "kenapa langit berwarna biru",
    ]
    prefixes = ["", "tolong jawab", "saya mau tahu"]

    rows = []
    for prefix in prefixes:
        for question in questions:
            text = f"{prefix} {question}".strip()
            rows.append({"text": text, "label": "general"})
            if len(rows) >= limit:
                return rows
    return rows


def contrastive_training_samples() -> list[dict[str, str]]:
    samples = {
        "greeting": [
            "permisi ada admin yang bisa bantu",
            "misi boleh dibantu sebentar",
            "misi ada admin yang sedang online",
            "permisi ada yang bisa membantu saya",
            "misi kak ada orang",
            "halo min ada yang bisa bantu",
            "halo ada orang di sini",
            "hai kak boleh ngobrol sebentar",
            "pagi min saya baru datang",
            "sore admin boleh minta bantuan",
            "malam kak masih online",
            "punten min saya mau bertanya dulu",
        ],
        "price_promo": [
            "kalau uang saya satu juta dapat pilihan apa saja",
            "kalau dana sekitar satu jutaan pilihannya apa",
            "dengan dana terbatas opsi produknya apa saja",
            "uang yang tersedia cuma sejuta bisa dapat apa",
            "opsi barang untuk dana dua juta apa saja",
            "budget sekitar dua juta ada barang apa",
            "dana 500 ribu bisa dapat produk apa",
            "maksimal anggaran saya 750rb ada pilihan apa",
            "produk under 1 jt tersedia apa saja",
            "dengan budget 3 juta bisa beli apa",
            "kisaran satu jutaan ada robot apa",
            "barang di bawah 2 juta apa saja",
            "kalau modal 900 ribu dapat apa",
            "daftar produk yang sedang diskon apa saja",
            "promo chogokin bulan ini berapa",
            "potongan harga yang aktif sekarang apa",
            "sale robot jadul hari ini apa saja",
            "cashback pembelian sekarang ada berapa",
            "harga setelah diskon menjadi berapa",
        ],
        "product_detail": [
            "material robot ini full die cast atau plastik",
            "bahan produknya seluruhnya diecast tidak",
            "figure ini menggunakan banyak part die cast",
            "komposisi bahannya metal atau pvc",
            "bagian badan robot terbuat dari besi tidak",
            "berapa persen material metal pada produk ini",
            "produknya full metal atau ada bagian plastik",
            "detail bahan figure ini apa saja",
            "body robot memakai diecast tidak",
            "apakah materialnya dominan logam",
        ],
        "shipping_transaction": [
            "pengiriman barang ini memakai asuransi tidak",
            "apakah barang ini dapat diasuransikan",
            "bisa pakai asuransi untuk barang ini",
            "unit ini bisa ditambah asuransi pengiriman tidak",
            "barang yang dibeli mendapat opsi asuransi tidak",
            "produk mahal bisa diberi proteksi kirim",
            "asuransi paket tersedia saat checkout tidak",
            "barang koleksi bisa diasuransikan selama pengiriman",
            "cara menambah proteksi untuk paket",
            "biaya asuransi pengiriman berapa",
            "paket dikirim dengan perlindungan kerusakan tidak",
            "opsi proteksi barang muncul saat pembayaran tidak",
            "bagaimana cara membeli barang ini",
            "langkah untuk beli produk ini seperti apa",
            "cara melakukan pembelian di toko ini",
            "kalau mau membeli produk harus mulai dari mana",
            "saya ingin beli barang ini caranya bagaimana",
            "panduan pesan produk sampai pembayaran",
            "bagaimana alur order barang di website",
            "cara memesan salah satu produk yang tampil",
            "kalau tertarik produk ini belinya lewat mana",
            "mau order robot ini langkahnya apa saja",
            "proses pembelian sampai barang dikirim bagaimana",
            "tolong jelaskan cara pesan produk dari awal",
        ],
        "general": [
            "sekarang menunjukkan pukul berapa",
            "pukul berapa saat ini",
            "boleh tahu sekarang jam berapa",
            "waktu saat ini menunjukkan jam berapa",
            "boleh kasih tahu waktu saat ini",
            "hari dan tanggal sekarang apa",
            "bantu saya membuat aplikasi sederhana",
            "bisa bantu buatkan sebuah program",
            "tolong bantu bikin program sederhana",
            "saya butuh bantuan membuat program",
            "tolong tuliskan program komputer",
            "ajarkan saya pemrograman dasar",
            "siapa pemimpin negara sekarang",
            "besok cuacanya seperti apa",
            "bantu jawab pertanyaan pelajaran",
            "saya mau tanya hal di luar toko",
        ],
        "compare": [
            "dari dua barang tadi yang unggul yang mana",
            "produk nomor satu dibanding nomor dua bagus mana",
            "kalau dua pilihan sebelumnya dibandingkan siapa menang",
            "antara barang pertama dan kedua lebih murah mana",
            "figure kondisi loose lawan misb pilih mana",
            "versi tanpa box vs lengkap box lebih baik mana",
            "dari dua robot sebelumnya yang lebih cocok untuk koleksi",
            "pilihan pertama dan pilihan kedua beda kualitas tidak",
            "coba komparasikan dua produk yang baru ditampilkan",
            "mana pemenang dari dua item tadi",
        ],
        "shipment_tracking": [
            "paket jne saya saat ini berada di kota apa",
            "kiriman jne sekarang sudah ada di kota mana",
            "posisi paket dari jne berada di mana",
            "paket kurir saya sudah sampai daerah mana",
            "kiriman jnt sudah bergerak ke mana",
            "barang yang dibawa sicepat posisinya di mana",
            "nomor resi ini dapat dilacak atau tidak",
            "apakah resi dari toko sudah bisa dilacak",
            "nomor tracking ini bisa dicek di sini tidak",
            "resi pengiriman belum dapat saya lacak",
            "cara melihat perjalanan paket dari resi",
            "resi yang diberikan toko belum muncul pergerakan",
            "paket sudah masuk kurir tetapi lokasinya belum jelas",
            "cek lokasi terbaru kiriman anteraja saya",
            "barang sedang transit di kota mana",
            "tolong lihat update perjalanan paket terakhir",
        ],
    }

    return [
        {"text": text, "label": label}
        for label, texts in samples.items()
        for text in texts
    ]


def hard_test_additions() -> list[dict[str, str]]:
    return [
        {"text": "sekarang pukul berapa ya", "label": "general"},
        {"text": "tolong kasih prakiraan cuaca besok", "label": "general"},
        {"text": "siapa kepala negara indonesia saat ini", "label": "general"},
        {"text": "bantu buat program sederhana dong", "label": "general"},
        {"text": "tokonya buka setiap hari atau tidak", "label": "general"},
        {"text": "admin robot jadul bisa dihubungi lewat apa", "label": "general"},
        {"text": "robot jadul itu toko khusus apa", "label": "general"},
        {"text": "ada cabang toko di luar jakarta tidak", "label": "general"},
        {"text": "gx 47 sama gx 48 beda apa", "label": "compare"},
        {"text": "kalau voltes dan daimos enakan pilih mana", "label": "compare"},
        {"text": "tolong komparasi grendizer lawan mazinger", "label": "compare"},
        {"text": "antara versi bandai dan takara lebih oke mana", "label": "compare"},
        {"text": "figure loose vs misb lebih worth it yang mana", "label": "compare"},
        {"text": "dari dua robot tadi mana yang lebih murah", "label": "compare"},
        {"text": "produk pertama dibanding kedua unggul mana", "label": "compare"},
        {"text": "mau tahu perbedaan getter one dan getter two", "label": "compare"},
        {"text": "resi saya JX9988776655 sudah bergerak belum", "label": "shipment_tracking"},
        {"text": "paket jne saya sekarang ada di kota mana", "label": "shipment_tracking"},
        {"text": "tolong tracking kiriman dengan nomor SC88776655", "label": "shipment_tracking"},
        {"text": "barang sudah diserahkan ke kurir tapi belum sampai", "label": "shipment_tracking"},
        {"text": "nomor resinya bisa dilacak dari sini tidak", "label": "shipment_tracking"},
        {"text": "update terakhir perjalanan paket saya apa", "label": "shipment_tracking"},
        {"text": "paket jnt saya kenapa tidak bergerak", "label": "shipment_tracking"},
        {"text": "cek posisi kiriman POS77665544", "label": "shipment_tracking"},
    ]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)


def count_labels(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = Counter(row["label"] for row in rows)
    return {label: counts.get(label, 0) for label in CANONICAL_LABELS}


def validate_training(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = count_labels(rows)
    missing = [label for label, count in counts.items() if count == 0]
    too_small = [
        label for label, count in counts.items() if count < MIN_TRAIN_ROWS_PER_LABEL
    ]

    if missing:
        raise ValueError(f"Intent tanpa data: {missing}")
    if too_small:
        raise ValueError(
            f"Intent di bawah {MIN_TRAIN_ROWS_PER_LABEL} baris: {too_small}"
        )
    return counts


def main() -> None:
    base_rows = read_repaired_csv(BASE_DATASET)
    addon_rows = read_repaired_csv(NOISE_ADDON)

    # Semua general pada addon lama adalah sapaan murni. General pada dataset
    # utama tetap dipertahankan karena berisi informasi umum toko.
    for row in addon_rows:
        if row["label"] == "general":
            row["label"] = "greeting"

    training_rows: list[dict[str, str]] = []
    training_seen: dict[str, str] = {}
    source_counts = {
        "base": add_unique_rows(training_rows, training_seen, base_rows),
        "noise_addon": add_unique_rows(training_rows, training_seen, addon_rows),
        "generated_greeting": add_unique_rows(
            training_rows, training_seen, generated_greetings()
        ),
        "generated_compare": add_unique_rows(
            training_rows, training_seen, generated_compare_samples()
        ),
        "generated_tracking": add_unique_rows(
            training_rows, training_seen, generated_tracking_samples()
        ),
        "generated_general": add_unique_rows(
            training_rows, training_seen, generated_out_of_scope_general()
        ),
        "contrastive": add_unique_rows(
            training_rows, training_seen, contrastive_training_samples()
        ),
    }

    training_counts = validate_training(training_rows)

    hard_rows = read_repaired_csv(LEGACY_HARD_TEST)
    hard_text_replacements = {
        "nomor pesanan 6864 progresnya gimana":
            "boleh cek perkembangan pesanan nomor 97531",
        "order #8123 sudah dikirim belum":
            "status proses order 86420 saat ini bagaimana",
        "yang promo tapi masih worth it apa":
            "diskon untuk produk chogokin bulan ini berapa",
        "cara checkout sampai bayar gimana":
            "cara membeli produk ini gimana",
    }
    for row in hard_rows:
        row["text"] = hard_text_replacements.get(row["text"], row["text"])
        if row["label"] == "general":
            row["label"] = "greeting"

    hard_output: list[dict[str, str]] = []
    hard_seen: dict[str, str] = {}
    add_unique_rows(hard_output, hard_seen, hard_rows)
    add_unique_rows(hard_output, hard_seen, hard_test_additions())

    hard_counts = count_labels(hard_output)
    invalid_hard_counts = {
        label: count for label, count in hard_counts.items() if count != 8
    }
    if invalid_hard_counts:
        raise ValueError(
            f"Hard test harus berisi 8 baris per intent: {invalid_hard_counts}"
        )

    overlap = sorted(set(training_seen).intersection(hard_seen))
    if overlap:
        raise ValueError(
            "Training dan hard test memiliki "
            f"{len(overlap)} teks yang sama: {overlap[:5]}"
        )

    write_csv(TRAIN_OUTPUT, training_rows)
    write_csv(HARD_TEST_OUTPUT, hard_output)

    report = {
        "contract_version": 1,
        "labels": list(CANONICAL_LABELS),
        "training_rows": len(training_rows),
        "training_counts": training_counts,
        "hard_test_rows": len(hard_output),
        "hard_test_counts": hard_counts,
        "train_hard_exact_overlap": len(overlap),
        "source_rows_added": source_counts,
        "outputs": {
            "training": str(TRAIN_OUTPUT),
            "hard_test": str(HARD_TEST_OUTPUT),
        },
    }
    REPORT_OUTPUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
