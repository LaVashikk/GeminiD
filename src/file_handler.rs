use anyhow::{anyhow, Result};
use base64::Engine;
use eframe::egui::{self, vec2, Color32, RichText, Stroke};
use gemini_rust::{prelude::*, Blob, FileState, Part};
use image::{ImageFormat, ImageReader};
use std::collections::HashMap;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};

static GLOBAL_FILE_CACHE: LazyLock<Mutex<HashMap<PathBuf, gemini_rust::File>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

// Supported Gemini MIME types
const GEMINI_MIME: &[&str] = &[
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/heic",
    "image/heif",
    "video/mp4",
    "video/mpeg",
    "video/quicktime",
    "video/avi",
    "video/x-flv",
    "video/mpg",
    "video/webm",
    "video/wmv",
    "video/3gpp",
    "audio/wav",
    "audio/mp3",
    "audio/aiff",
    "audio/aac",
    "audio/ogg",
    "audio/flac",
    "text/plain",
    "text/html",
    "text/css",
    "text/javascript",
    "text/typescript",
    "application/x-javascript",
    "application/json",
    "text/xml",
    "application/rtf",
    "text/rtf",
    "application/pdf",
];

#[derive(Debug, Default, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum AttachmentState {
    #[default]
    Local,
    Uploading,
    Uploaded(gemini_rust::File),
    Failed(String),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Attachment {
    pub path: PathBuf,
    pub mime: String,
    #[serde(skip)]
    pub state: AttachmentState,
}

impl Attachment {
    pub fn from_path(path: PathBuf) -> Self {
        let mime = mime_guess::from_path(&path)
            .first_or_octet_stream()
            .to_string();
        Self {
            path,
            mime,
            state: AttachmentState::Local,
        }
    }
}

/// Returns either a Part with inline data or a FileHandle of the uploaded file
pub enum FileResult {
    /// Inline data part for direct use
    InlinePart(Part),
    /// Handle of the uploaded file for use in the API
    UploadedFile(FileHandle),
}

pub async fn convert_file_to_part(
    client: &Gemini,
    path: &Path,
    upload: bool,
) -> Result<FileResult> {
    const MAX_INLINE_SIZE: u64 = 20 * 1024 * 1024; // 20 MB

    // Check file size first
    let metadata = tokio::fs::metadata(path).await?;
    if metadata.len() > MAX_INLINE_SIZE && !upload {
        return Err(anyhow!(
            "File is too large for inline transmission ({} bytes > 20MB limit). Please enable 'File API' in settings.",
            metadata.len()
        ));
    }

    if upload {
        if let Ok(cache) = GLOBAL_FILE_CACHE.lock() {
            if let Some(remote_file) = cache.get(path) {
                // Check expiration
                let is_expired = if let Some(exp) = remote_file.expiration_time {
                    exp < time::OffsetDateTime::now_utc()
                } else {
                    false
                };

                if !is_expired {
                    log::info!("Global cache hit for {}", path.display());
                    return Ok(FileResult::UploadedFile(
                        client.file_from_model(remote_file.clone()),
                    ));
                } else {
                    log::info!("Global cache expired for {}", path.display());
                }
            }
        }
    }

    // Read file into bytes asynchronously
    let file_bytes = tokio::fs::read(path).await?;

    let mime_type = mime_guess::from_path(path).first_or_octet_stream();
    let mut mime_str = mime_type.to_string();

    if (mime_str.starts_with("application") && mime_str != "application/pdf")
        || (mime_str.starts_with("text") && mime_str != "text/plain")
    {
        mime_str = "text/plain".to_string();
    }

    log::info!(
        "Processing file: {}, MIME type: {}",
        path.display(),
        mime_str
    );

    // Convert non-PNG/JPEG images to PNG
    let final_bytes = if mime_type.type_() == "image" {
        match ImageReader::new(Cursor::new(&file_bytes))
            .with_guessed_format()?
            .format()
        {
            Some(format) if !matches!(format, ImageFormat::Png | ImageFormat::Jpeg) => {
                log::debug!("Got {format:?} image, converting to png");
                mime_str = "image/png".to_string();

                let img = image::load_from_memory(&file_bytes)?;
                let mut buf = Vec::new();
                img.write_to(&mut Cursor::new(&mut buf), ImageFormat::Png)?;
                buf
            }
            _ => {
                // Already PNG/JPEG or unknown image format, send as is
                file_bytes
            }
        }
    } else {
        // Use source bytes for video, text, and other file types
        file_bytes
    };

    // Check MIME type support
    if !GEMINI_MIME.contains(&mime_str.as_str()) {
        return Err(anyhow!(
            "Unsupported MIME type: {}. Supported types are: {:?}",
            mime_str,
            GEMINI_MIME
        ));
    }

    if upload {
        log::info!("Uploading file...");

        let file_handle = client
            .create_file(final_bytes)
            .display_name(
                path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("uploaded_file"),
            )
            .with_mime_type(mime_str.parse()?)
            .upload()
            .await?;

        log::info!(
            "File uploaded: {}, waiting for processing...",
            file_handle.name()
        );

        let start_time = Instant::now();
        let timeout = Duration::from_secs(300);

        loop {
            let fresh_file_handle = client.get_file(file_handle.name()).await?;
            match fresh_file_handle.get_file_meta().state.clone() {
                Some(FileState::Active) => {
                    log::info!("File {} is ACTIVE and ready.", file_handle.name());

                    // Update cache
                    if let Ok(mut cache) = GLOBAL_FILE_CACHE.lock() {
                        cache.insert(
                            path.to_path_buf(),
                            fresh_file_handle.get_file_meta().clone(),
                        );
                    }

                    break;
                }
                Some(FileState::Failed) => {
                    return Err(anyhow!("File processing failed on Google server side."));
                }
                Some(FileState::Processing) | Some(FileState::StateUnspecified) | None => {
                    if start_time.elapsed() > timeout {
                        return Err(anyhow!("Timeout waiting for file to become ACTIVE"));
                    }
                    log::debug!("File still processing, waiting...");
                    tokio::time::sleep(Duration::from_secs(2)).await;
                }
                _ => {
                    if start_time.elapsed() > timeout {
                        return Err(anyhow!("Timeout or unknown state"));
                    }
                    tokio::time::sleep(Duration::from_secs(2)).await;
                }
            }
        }
        // ---------------------------------------

        // Return file handle only when it is ACTIVE
        Ok(FileResult::UploadedFile(file_handle))
    } else {
        let base64 = base64::engine::general_purpose::STANDARD.encode(&final_bytes);
        log::debug!(
            "Converted file to {} bytes of base64 with mime type {}",
            base64.len(),
            mime_str
        );

        let blob = Blob::new(mime_str, base64);
        let part = Part::InlineData {
            inline_data: blob,
            media_resolution: None,
        };

        Ok(FileResult::InlinePart(part))
    }
}

pub fn show_files(ui: &mut egui::Ui, files: &mut Vec<Attachment>, mutate: bool) {
    const MAX_PREVIEW_HEIGHT: f32 = 128.0;
    let pointer_pos = ui.input(|i| i.pointer.interact_pos());
    let mut showing_x = false;

    files.retain_mut(|file| {
        let file_path = &mut file.path;
        let path_string = file_path.display().to_string();
        // let mime_type = mime_guess::from_path(&file_path).first_or_octet_stream();
        // todo: Use cached mime?
        let mime_type = &file.mime;

        let is_exist = file_path.exists();
        let frame_color = match file.state {
            AttachmentState::Local => if is_exist { egui::Color32::GRAY } else { egui::Color32::from_rgb(201, 178, 141) },
            AttachmentState::Uploading => egui::Color32::from_rgb(141, 164, 201),
            AttachmentState::Uploaded(_) => egui::Color32::from_rgb(141, 189, 156),
            AttachmentState::Failed(_) => egui::Color32::from_rgb(201, 141, 141),
        };

        let custom_frame =
            egui::Frame::group(ui.style()).stroke(egui::Stroke::new(1.0, frame_color));

        let resp = custom_frame
            .show(ui, |ui| {
                ui.vertical(|ui| {
                    // Display preview or icon depending on the file type
                    match mime_type.split('/').next().unwrap_or("") {
                        "image" if is_exist => {
                            ui.add(
                                egui::Image::new(format!("file://{path_string}"))
                                    .max_height(MAX_PREVIEW_HEIGHT)
                                    .fit_to_original_size(1.0),
                            );
                        }
                        _ => {
                            // Create a container-frame with a fixed height
                            egui::Frame::NONE.show(ui, |ui| {
                                // Force the height to be equal to the maximum preview image height
                                ui.set_height(MAX_PREVIEW_HEIGHT);
                                // Set the width so that the widget is not too narrow
                                ui.set_width(MAX_PREVIEW_HEIGHT * 1.2);

                                // Center the icon inside this frame
                                ui.centered_and_justified(|ui| {
                                    let icon = if !is_exist {
                                        "⚠"
                                    } else {
                                        match mime_type.split('/').next().unwrap_or("") {
                                            "video" => "🎬",
                                            "audio" => "🎶",
                                            // "text" => "",
                                            _ => "📎",
                                        }
                                    };

                                    ui.label(RichText::new(icon).size(40.0));
                                });
                            });
                        }
                    }

                    let mut text = file_path.file_name().unwrap_or_default().to_string_lossy();
                    if !is_exist {
                        text.to_mut().push_str(" (FILE NOT FOUND)");
                    }
                    ui.add(egui::Label::new(RichText::new(text).small()).truncate());

                    if let AttachmentState::Failed(err) = &file.state {
                         ui.colored_label(Color32::RED, "Failed");
                         ui.label(RichText::new(err).small().color(Color32::RED));
                    } else if matches!(file.state, AttachmentState::Uploading) {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label("Uploading...");
                        });
                    }
                });
            })
            .response;

        let interact_resp = ui
            .interact(resp.rect, resp.id.with("interact"), egui::Sense::click())
            .on_hover_text(&path_string);
        if interact_resp.hovered() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
        }
        if !mutate && interact_resp.clicked() {
            if is_exist {
                if let Err(e) = open::that(&mut *file_path) {
                    log::error!("Failed to open file {}: {}", file_path.display(), e);
                }
            }
        }

        if !mutate || showing_x {
            return true;
        }

        if let Some(pos) = pointer_pos {
            if resp.rect.expand(8.0).contains(pos) {
                showing_x = true;

                // render an ❌ in a red circle
                let top = resp.rect.right_top();
                let x_rect = egui::Rect::from_center_size(top, vec2(16.0, 16.0));
                let contains_pointer = x_rect.contains(pos);

                ui.painter()
                    .circle_filled(top, 10.0, ui.visuals().window_fill);
                ui.painter().circle_filled(
                    top,
                    8.0,
                    if contains_pointer {
                        ui.visuals().gray_out(ui.visuals().error_fg_color)
                    } else {
                        ui.visuals().error_fg_color
                    },
                );
                ui.painter().line_segment(
                    [top - vec2(3.0, 3.0), top + vec2(3.0, 3.0)],
                    Stroke::new(2.0, Color32::WHITE),
                );
                ui.painter().line_segment(
                    [top - vec2(3.0, -3.0), top + vec2(3.0, -3.0)],
                    Stroke::new(2.0, Color32::WHITE),
                );

                if contains_pointer && ui.input(|i| i.pointer.primary_clicked()) {
                    return false;
                }
            }
        }

        true
    });
}
