#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$ROOT_DIR/datasets"
EXT_DIR="$DATA_DIR/external"
STATUS_FILE="$EXT_DIR/DOWNLOAD_STATUS.txt"

mkdir -p "$DATA_DIR" "$EXT_DIR"

log() {
	echo "$1" | tee -a "$STATUS_FILE"
}

download_with_fallback() {
	local name="$1"
	local out="$2"
	shift 2
	local urls=("$@")

	for url in "${urls[@]}"; do
		log "Trying $name: $url"
		if curl -L --fail --show-error --silent --retry 2 --retry-delay 2 --connect-timeout 12 --max-time 60 "$url" -o "$out"; then
			log "SUCCESS $name -> $out"
			return 0
		fi
	done
	rm -f "$out"
	log "FAILED $name (all URL attempts failed)"
	return 1
}

download_sample_video() {
	local out="$DATA_DIR/sample_drive.mp4"
	if [[ -s "$out" ]]; then
		log "SKIP sample driving video (already present): $out"
		return 0
	fi
	download_with_fallback "sample driving video" "$out" \
		"https://github.com/intel-iot-devkit/sample-videos/raw/master/car-detection.mp4" \
		"https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/cv/tracking/vtest.avi"
	echo "Downloaded from script fallback list" > "$DATA_DIR/DOWNLOADED_FROM.txt"
}

download_kitti_devkit() {
	local out="$EXT_DIR/kitti_devkit_object.zip"
	if [[ -s "$out" ]]; then
		log "SKIP KITTI devkit (already present): $out"
		return 0
	fi
	download_with_fallback "KITTI devkit" "$out" \
		"https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_object.zip"
}

download_nuscenes_mini() {
	local out="$EXT_DIR/nuscenes_v1.0-mini.tgz"
	if [[ -d "$EXT_DIR/extracted/nuscenes/v1.0-mini" ]]; then
		log "SKIP nuScenes mini (already extracted): $EXT_DIR/extracted/nuscenes/v1.0-mini"
		return 0
	fi
	if [[ -s "$out" ]] && tar -tzf "$out" >/dev/null 2>&1; then
		log "SKIP nuScenes mini (existing archive verified): $out"
		return 0
	fi
	download_with_fallback "nuScenes mini" "$out" \
		"https://www.nuscenes.org/data/v1.0-mini.tgz"
	if tar -tzf "$out" >/dev/null 2>&1; then
		log "VERIFIED nuScenes mini archive"
	else
		rm -f "$out"
		log "FAILED nuScenes mini integrity check"
		return 1
	fi
}

download_bdd100k_with_fallback() {
	local out="$EXT_DIR/bdd100k_labels_release.zip"

	if [[ -s "$out" ]]; then
		log "SKIP BDD100K labels (already present): $out"
		return 0
	fi

	if download_with_fallback "BDD100K labels" "$out" \
		"https://bdd-data.berkeley.edu/data/bdd100k_labels_release.zip"; then
		return 0
	fi

	if command -v kaggle >/dev/null 2>&1; then
		local raw_ids="${KAGGLE_BDD100K_DATASETS:-solesensei/solesensei_bdd100k,shreayan98c/bdd100k,solesensei/bdd100k}"
		local old_ifs="$IFS"
		IFS=','
		read -r -a dataset_ids <<< "$raw_ids"
		IFS="$old_ifs"

		for kaggle_dataset in "${dataset_ids[@]}"; do
			kaggle_dataset="$(echo "$kaggle_dataset" | xargs)"
			[[ -z "$kaggle_dataset" ]] && continue
			log "Trying BDD100K via Kaggle API dataset=$kaggle_dataset"
			if kaggle datasets download -d "$kaggle_dataset" -p "$EXT_DIR" --force; then
				log "SUCCESS BDD100K via Kaggle API dataset=$kaggle_dataset"
				return 0
			fi
		done

		log "FAILED BDD100K via Kaggle API (all dataset IDs failed)"
	else
		log "Kaggle CLI not found; skipping Kaggle fallback"
	fi

	return 1
}

extract_if_present() {
	mkdir -p "$EXT_DIR/extracted"

	if [[ -f "$EXT_DIR/kitti_devkit_object.zip" ]]; then
		if [[ ! -d "$EXT_DIR/extracted/kitti_devkit_object/devkit_object" ]]; then
			mkdir -p "$EXT_DIR/extracted/kitti_devkit_object"
			unzip -oq "$EXT_DIR/kitti_devkit_object.zip" -d "$EXT_DIR/extracted/kitti_devkit_object" || true
		else
			log "SKIP KITTI extraction (already extracted)"
		fi
	fi

	if [[ -f "$EXT_DIR/nuscenes_v1.0-mini.tgz" ]]; then
		if [[ -d "$EXT_DIR/extracted/nuscenes/v1.0-mini" ]]; then
			log "SKIP nuScenes extraction (already extracted)"
		elif tar -tzf "$EXT_DIR/nuscenes_v1.0-mini.tgz" >/dev/null 2>&1; then
			mkdir -p "$EXT_DIR/extracted/nuscenes"
			tar -xzf "$EXT_DIR/nuscenes_v1.0-mini.tgz" -C "$EXT_DIR/extracted/nuscenes"
		else
			log "SKIP nuScenes extraction (archive failed integrity check)"
		fi
	fi
}

main() {
	: > "$STATUS_FILE"
	log "Dataset download started"

	download_sample_video || true
	download_kitti_devkit || true
	download_nuscenes_mini || true
	download_bdd100k_with_fallback || true
	extract_if_present || true

	log "Dataset download finished"
	log "Manual acceptance pages:"
	log "KITTI: https://www.cvlibs.net/datasets/kitti/"
	log "BDD100K: https://bdd-data.berkeley.edu/"
	log "nuScenes: https://www.nuscenes.org/"

	echo
	echo "Artifacts overview:"
	ls -lh "$DATA_DIR" "$EXT_DIR" 2>/dev/null || true
	echo
	echo "Status log: $STATUS_FILE"
}

main "$@"
