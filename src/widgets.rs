use std::fmt;

use chrono::{Datelike, Local};
use eframe::{
    egui::{self, collapsing_header::CollapsingState, CornerRadius, Frame, Layout, Stroke, Vec2},
    emath::Numeric,
};
use egui_modal::{Icon, Modal};
use gemini_rust::{Gemini, GeminiBuilder, GenerationConfig, Model, ThinkingConfig};
use reqwest;
use serde::{Deserialize, Serialize};

#[derive(Default, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct ModelPicker {
    pub selected: GeminiModel,
    settings: ModelSettings,
    pub system_prompt: Option<String>,
}

pub enum RequestInfoType {
    LoadSettings,
    LoginGoogle,
    LogoutGoogle,
    SelectProject(String),
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuthMethod {
    #[default]
    ApiKey,
    CodeAssist,
}

impl fmt::Display for AuthMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AuthMethod::ApiKey => write!(f, "API Key"),
            AuthMethod::CodeAssist => write!(f, "Google Code Assist"),
        }
    }
}

/// Represents the available Gemini models.
#[derive(
    Default, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, enum_iterator::Sequence,
)]
pub enum GeminiModel {
    #[default]
    #[serde(rename = "gemini-3-flash-preview")]
    Gemini30Flash,
    #[serde(rename = "gemini-3-pro-preview")]
    Gemini30Pro,
    #[serde(rename = "gemini-2.0-flash")]
    Gemini20Flash,

    #[serde(rename = "gemini-2.0-flash-lite")]
    Gemini20FlashLite,

    #[serde(rename = "gemini-2.5-pro")]
    Gemini25Pro,

    #[serde(rename = "gemini-2.5-flash")]
    Gemini25Flash,

    #[serde(rename = "gemini-1.5-flash")]
    Gemini15Flash,

    #[serde(rename = "gemini-1.5-flash-8b")]
    Gemini15Flash8b,

    #[serde(rename = "gemini-2.5-flash-preview-05-20")]
    Gemini25FlashPreview0520,

    #[serde(rename = "gemini-2.0-flash-thinking-exp-01-21")]
    Gemini20FlashThinkingExp0121,

    #[serde(rename = "gemini-2.0-flash-thinking-exp-1219")]
    Gemini20FlashThinkingExp1219,

    #[serde(rename = "gemma-3-1b-it")]
    Gemma31bIt,

    #[serde(rename = "gemma-3-4b-it")]
    Gemma34bIt,

    #[serde(rename = "gemma-3-12b-it")]
    Gemma312bIt,

    #[serde(rename = "gemma-3-27b-it")]
    Gemma327bIt,

    #[serde(rename = "gemma-3n-e4b-it")]
    Gemma3nE4bIt,

    #[serde(rename = "gemma-3n-e2b-it")]
    Gemma3nE2bIt,

    // Models for paid quota
    #[serde(rename = "gemini-1.5-pro")]
    Gemini15Pro,

    #[serde(rename = "gemini-2.5-pro-preview-03-25")]
    Gemini25ProPreview0325,

    #[serde(rename = "gemini-2.5-pro-preview-05-06")]
    Gemini25ProPreview0506,

    #[serde(rename = "gemini-2.5-pro-preview-06-05")]
    Gemini25ProPreview0605,
    // To add a new model, simply add a new variant here
    // and its corresponding `rename` attribute.
    // #[serde(rename = "new-model-name")]
    // NewModelName,
}

impl From<GeminiModel> for Model {
    fn from(val: GeminiModel) -> Self {
        let model_id = serde_json::to_value(val)
            .expect("Failed to serialize model enum")
            .as_str()
            .expect("Model enum should serialize to string")
            .to_string();

        Model::Custom(format!("models/{}", model_id))
    }
}

/// Allows the enum to be printed or converted to its string representation.
impl fmt::Display for GeminiModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            serde_json::to_string(self)
                .expect("Failed to serialize model")
                .trim_matches('"')
        )
    }
}

fn collapsing_frame<R>(
    ui: &mut egui::Ui,
    heading: &str,
    show: impl FnOnce(&mut egui::Ui) -> R,
) -> egui::Response {
    let style = ui.style();

    egui::Frame {
        inner_margin: egui::Margin::same(4),
        corner_radius: style.visuals.menu_corner_radius,
        fill: style.visuals.extreme_bg_color,
        ..egui::Frame::NONE
    }
    .show(ui, |ui| {
        ui.with_layout(Layout::top_down_justified(egui::Align::Min), |ui| {
            let mut state = CollapsingState::load_with_default_open(
                ui.ctx(),
                ui.make_persistent_id(heading),
                false,
            );

            let resp = ui.add(
                egui::Label::new(heading)
                    .selectable(false)
                    .sense(egui::Sense::click()),
            );
            if resp.clicked() {
                state.toggle(ui);
            }

            state.show_body_unindented(ui, |ui| {
                ui.separator();
                ui.vertical(|ui| {
                    show(ui);
                });
            });

            resp
        });
    })
    .response
}

const TEMPLATE_HINT_TEXT: &str =
    "A system prompt for the model. E.g., 'You are a helpful assistant that specializes in writing Rust code.'";

impl ModelPicker {
    pub fn create_client(
        &self,
        api_key: &str,
        proxy_path: Option<String>,
    ) -> Result<Gemini, gemini_rust::ClientError> {
        let mut client_builder = reqwest::Client::builder();

        if let Some(proxy_url) = proxy_path {
            if !proxy_url.is_empty() {
                if let Ok(proxy) = reqwest::Proxy::all(proxy_url) {
                    client_builder = client_builder.proxy(proxy);
                } else {
                    log::error!("Invalid proxy URL, ignoring it.");
                }
            }
        }

        GeminiBuilder::new(api_key)
            .with_model(Model::from(self.selected))
            .with_http_client(client_builder)
            .build()
    }

    pub fn show<R>(&mut self, ui: &mut egui::Ui, _request_info: &mut R)
    where
        R: FnMut(RequestInfoType),
    {
        egui::ComboBox::from_id_salt("model_selector_combobox")
            .selected_text(self.selected.to_string())
            .show_ui(ui, |ui| {
                for model in enum_iterator::all::<GeminiModel>() {
                    if ui
                        .selectable_label(self.selected == model, model.to_string())
                        .clicked()
                    {
                        self.selected = model;
                    }
                }
            });

        ui.collapsing("Inference Settings", |ui| {
            self.settings.show(ui);
        });

        collapsing_frame(ui, "System Prompt", |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label("A system prompt can be used to specify custom behavior from the model.");
            });

            let mut enabled = self.system_prompt.is_some();
            ui.horizontal(|ui| {
                ui.add(toggle(&mut enabled));
                ui.label("Enable custom system prompt");
            });
            if !enabled {
                self.system_prompt = None;
            } else if self.system_prompt.is_none() {
                self.system_prompt = Some(String::new());
            }

            ui.add_enabled_ui(self.system_prompt.is_some(), |ui| {
                if let Some(ref mut template) = self.system_prompt {
                    ui.add(
                        egui::TextEdit::multiline(template)
                            .hint_text(TEMPLATE_HINT_TEXT)
                            .desired_rows(3),
                    );
                }
            });
        });
    }

    #[inline]
    pub fn get_generation_config(&self) -> GenerationConfig {
        self.settings.clone().into()
    }
}

#[derive(Default, Clone, Deserialize, Serialize)]
#[serde(default)]
struct ModelSettings {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub num_predict: Option<i32>, // Mapped to maxOutputTokens
    pub stop: Option<Vec<String>>,
    pub include_thoughts: bool,
    pub thinking_budget: Option<i32>,
}

impl From<ModelSettings> for GenerationConfig {
    fn from(value: ModelSettings) -> Self {
        let mut config = GenerationConfig::default();
        config.temperature = value.temperature;
        config.top_p = value.top_p;
        config.top_k = value.top_k.map(|k| k as i32);
        config.max_output_tokens = value.num_predict;
        config.stop_sequences = value.stop;

        if value.include_thoughts || value.thinking_budget.is_some() {
            let mut thinking_config = ThinkingConfig::default();
            if value.include_thoughts {
                thinking_config.include_thoughts = Some(true);
            }
            if let Some(budget) = value.thinking_budget {
                thinking_config.thinking_budget = Some(budget);
            }
            config.thinking_config = Some(thinking_config);
        }
        config
    }
}

impl ModelSettings {
    fn edit_numeric<N: Numeric>(
        ui: &mut egui::Ui,
        val: &mut Option<N>,
        mut default: N,
        speed: f64,
        range: std::ops::RangeInclusive<N>,
        name: &str,
        doc: &str,
    ) {
        collapsing_frame(ui, name, |ui: &mut egui::Ui| {
            ui.label(doc);
            let mut enabled = val.is_some();
            ui.horizontal(|ui| {
                ui.add(toggle(&mut enabled));
                ui.label("Enable");
            });

            if !enabled {
                *val = None;
            } else if val.is_none() {
                *val = Some(default);
            }

            ui.add_enabled_ui(val.is_some(), |ui| {
                ui.horizontal(|ui| {
                    if let Some(val) = val {
                        ui.add(egui::DragValue::new(val).speed(speed).range(range));
                    } else {
                        ui.add(
                            egui::DragValue::new(&mut default)
                                .speed(speed)
                                .range(range.clone()),
                        );
                    }
                    if ui
                        .button("reset")
                        .on_hover_text("Reset to default")
                        .clicked()
                    {
                        *val = None;
                    }
                });
            });
        });
    }

    fn show(&mut self, ui: &mut egui::Ui) {
        if ui.button("Reset Settings").clicked() {
            *self = Self::default();
        }

        collapsing_frame(ui, "Thinking", |ui| {
            ui.label("Enable native thinking for Gemini 2.5 models to improve reasoning.");
            ui.checkbox(&mut self.include_thoughts, "Include thought summaries");

            ui.add_enabled_ui(self.include_thoughts, |ui| {
                let mut budget_enabled = self.thinking_budget.is_some();
                ui.horizontal(|ui| {
                    ui.add(toggle(&mut budget_enabled));
                    ui.label("Set thinking budget");
                });

                if !budget_enabled {
                    self.thinking_budget = None;
                } else if self.thinking_budget.is_none() {
                    self.thinking_budget = Some(-1); // -1 for dynamic budget
                }

                if let Some(ref mut budget) = self.thinking_budget {
                    ui.add(egui::DragValue::new(budget).speed(100.0).range(-1..=32768))
                        .on_hover_text("Token budget for thinking. -1 for dynamic, 0 to disable.");
                }
            });
        });

        Self::edit_numeric(ui, &mut self.temperature, 0.9, 0.01, 0.0..=1.0, "Temperature", "Controls the randomness of the output. Higher values (e.g., 1.0) produce more creative responses, while lower values (e.g., 0.2) make the output more deterministic.");
        Self::edit_numeric(
            ui,
            &mut self.num_predict,
            2048,
            1.0,
            1..=8192,
            "Max Output Tokens",
            "Maximum number of tokens to generate in the response.",
        );
        Self::edit_numeric(ui, &mut self.top_k, 40, 1.0, 1..=100, "Top-K", "Changes how the model selects tokens for output. A lower value limits the sampling to a smaller set of the most likely tokens.");
        Self::edit_numeric(ui, &mut self.top_p, 0.95, 0.01, 0.0..=1.0, "Top-P", "Changes how the model selects tokens for output, sampling from a cumulative probability distribution. Use either Top-K or Top-P, not both.");

        collapsing_frame(ui, "Stop Sequence", |ui| {
            ui.label("A set of up to 5 character sequences that will stop output generation.");
            let mut enabled = self.stop.is_some();

            ui.horizontal(|ui| {
                ui.add(toggle(&mut enabled));
                ui.label("Enable");
            });

            if !enabled {
                self.stop = None;
            } else if self.stop.is_none() {
                self.stop = Some(Vec::new());
            }

            ui.add_enabled_ui(self.stop.is_some(), |ui| {
                if let Some(ref mut stop) = self.stop {
                    stop.retain_mut(|pat| {
                        let mut keep = true;
                        ui.horizontal(|ui| {
                            ui.text_edit_singleline(pat);
                            if ui.button("❌").clicked() {
                                keep = false;
                            }
                        });
                        keep
                    });
                    if stop.len() < 5 && ui.button("➕ Add").clicked() {
                        stop.push(String::new());
                    }
                    if ui.button("Clear").clicked() {
                        stop.clear();
                    }
                }
            });
        });
    }
}

pub fn centerer(ui: &mut egui::Ui, add_contents: impl FnOnce(&mut egui::Ui)) {
    let available_height = ui.available_height();
    ui.horizontal(|ui| {
        let id = ui.id().with("_centerer");
        let last_size: Option<(f32, f32)> = ui.memory_mut(|mem| mem.data.get_temp(id));
        if let Some(last_size) = last_size {
            ui.add_space((ui.available_width() - last_size.0) / 2.0);
        }

        let res = ui
            .vertical(|ui| {
                if let Some(last_size) = last_size {
                    ui.add_space((available_height - last_size.1) / 2.0)
                }
                ui.scope(|ui| {
                    add_contents(ui);
                })
                .response
            })
            .inner;

        let (width, height) = (res.rect.width(), res.rect.height());
        ui.memory_mut(|mem| mem.data.insert_temp(id, (width, height)));

        match last_size {
            None => ui.ctx().request_repaint(),
            Some((last_width, last_height)) if last_width != width || last_height != height => {
                ui.ctx().request_repaint()
            }
            Some(_) => {}
        }
    });
}

pub fn suggestion(ui: &mut egui::Ui, text: &str, subtext: &str) -> egui::Response {
    let mut resp = Frame::group(ui.style())
        .corner_radius(CornerRadius::same(6))
        .stroke(Stroke::NONE)
        .fill(ui.style().visuals.faint_bg_color)
        .show(ui, |ui| {
            ui.vertical(|ui| {
                ui.add(egui::Label::new(text).selectable(false));
                ui.add_enabled(false, egui::Label::new(subtext).selectable(false));
            });
            ui.add_space(ui.available_width());
        })
        .response;

    if resp.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }

    // for some reason egui sets `Frame::group` to not sense clicks, so we
    // have to hack it here
    if resp.hovered()
        && ui.input(|i| {
            i.pointer.any_click()
                && i.pointer
                    .interact_pos()
                    .map(|p| resp.rect.contains(p))
                    .unwrap_or(false)
        })
    {
        resp.flags.insert(egui::response::Flags::CLICKED);
    }

    resp
}

pub fn dummy(ui: &mut egui::Ui) {
    ui.add_sized(Vec2::ZERO, egui::Label::new("").selectable(false));
}

fn toggle_ui(ui: &mut egui::Ui, on: &mut bool) -> egui::Response {
    let desired_size = ui.spacing().interact_size.y * egui::vec2(2.0, 1.0);
    let (rect, mut response) = ui.allocate_exact_size(desired_size, egui::Sense::click());
    if response.clicked() {
        *on = !*on;
        response.mark_changed();
    }
    response.widget_info(|| {
        egui::WidgetInfo::selected(egui::WidgetType::Checkbox, *on, response.hovered(), "")
    });

    if ui.is_rect_visible(rect) {
        let how_on = ui.ctx().animate_bool(response.id, *on);
        let visuals = ui.style().interact_selectable(&response, *on);
        let rect = rect.expand(visuals.expansion);
        let radius = 0.5 * rect.height();
        ui.painter().rect(
            rect,
            radius,
            visuals.bg_fill,
            visuals.bg_stroke,
            egui::StrokeKind::Outside,
        );
        let circle_x = egui::lerp((rect.left() + radius)..=(rect.right() - radius), how_on);
        let center = egui::pos2(circle_x, rect.center().y);
        ui.painter()
            .circle(center, 0.75 * radius, visuals.bg_fill, visuals.fg_stroke);
    }

    response
}

#[inline]
fn toggle(on: &mut bool) -> impl egui::Widget + '_ {
    move |ui: &mut egui::Ui| toggle_ui(ui, on)
}

fn help(ui: &mut egui::Ui, text: &str, add_contents: impl FnOnce(&mut egui::Ui)) {
    ui.horizontal(|ui| {
        add_contents(ui);
        ui.add_enabled(false, egui::Label::new("(?)").selectable(false))
            .on_disabled_hover_text(text);
    });
}

// This is the main settings struct.
#[derive(Deserialize, Serialize, Clone)]
#[serde(default)]
pub struct Settings {
    pub auth_method: AuthMethod,
    pub api_key: String,
    pub oauth_token: String,
    pub project_id: String,
    #[serde(skip)]
    pub available_projects: Vec<String>,

    pub model_picker: ModelPicker,
    pub inherit_chat_picker: bool,
    pub use_streaming: bool,
    #[serde(default)]
    pub include_thoughts_in_history: bool,
    #[serde(default)]
    pub public_file_upload: bool,
    pub proxy_path: Option<String>,
    pub let_it_snow: bool,
    is_winter: bool,
}

impl Default for Settings {
    fn default() -> Self {
        let is_winter = matches!(Local::now().month(), 12 | 1 | 2);
        Self {
            auth_method: AuthMethod::ApiKey,
            api_key: String::new(), // todo try read from env
            oauth_token: String::new(),
            project_id: String::new(),
            available_projects: Vec::new(),
            model_picker: ModelPicker::default(),
            inherit_chat_picker: true,
            use_streaming: true,
            include_thoughts_in_history: false,
            public_file_upload: true,
            proxy_path: None,
            is_winter: is_winter,
            let_it_snow: is_winter,
        }
    }
}

impl Settings {
    pub fn show_modal(&mut self, modal: &Modal) {
        modal.show(|ui| {
            modal.title(ui, "Reset Settings");
            modal.frame(ui, |ui| {
                modal.body_and_icon(
                    ui,
                    "Are you sure you want to reset global settings? \
                    This action cannot be undone!",
                    Icon::Warning,
                );
            });
            modal.buttons(ui, |ui| {
                if modal.button(ui, "No").clicked() {
                    modal.close();
                }
                if modal.caution_button(ui, "Yes").clicked() {
                    *self = Self::default();
                    modal.close();
                }
            });
        });
    }

    async fn ask_save_settings(settings: Self) {
        let Some(file) = rfd::AsyncFileDialog::new()
            .add_filter("JSON file", &["json"])
            .save_file()
            .await
        else {
            log::warn!("no file selected");
            return;
        };

        let Ok(f) = std::fs::File::create(file.path())
            .map_err(|e| log::error!("failed to create file: {e}"))
        else {
            return;
        };

        let _ = serde_json::to_writer_pretty(f, &settings)
            .map_err(|e| log::error!("failed to save settings: {e}"));
    }

    pub fn show<R>(&mut self, ui: &mut egui::Ui, request_info: &mut R, modal: &Modal)
    where
        R: FnMut(RequestInfoType),
    {
        ui.heading("Authentication");
        egui::ComboBox::from_label("Method")
            .selected_text(self.auth_method.to_string())
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.auth_method, AuthMethod::ApiKey, "API Key");
                ui.selectable_value(
                    &mut self.auth_method,
                    AuthMethod::CodeAssist,
                    "Google Code Assist",
                );
            });

        ui.add_space(4.0);

        match self.auth_method {
            AuthMethod::ApiKey => {
                egui::Grid::new("settings_grid_api")
                    .num_columns(2)
                    .striped(true)
                    .min_row_height(32.0)
                    .show(ui, |ui| {
                        ui.label("API Key");
                        ui.add(
                            egui::TextEdit::singleline(&mut self.api_key)
                                .password(true)
                                .hint_text("Enter your Google AI Studio API Key"),
                        );
                        ui.end_row();
                    });
            }
            AuthMethod::CodeAssist => {
                ui.horizontal(|ui| {
                    if ui.button("🔑 Login with Google").clicked() {
                        request_info(RequestInfoType::LoginGoogle);
                    }
                    if !self.oauth_token.is_empty() {
                        ui.label("✅ Logged in");
                        if ui.button("🚪 Logout").clicked() {
                            request_info(RequestInfoType::LogoutGoogle);
                        }
                    }
                });

                if !self.available_projects.is_empty() {
                    ui.add_space(4.0);
                    egui::ComboBox::from_label("Project")
                        .selected_text(if self.project_id.is_empty() {
                            "Select Project"
                        } else {
                            &self.project_id
                        })
                        .show_ui(ui, |ui| {
                            for proj in &self.available_projects {
                                if ui
                                    .selectable_label(self.project_id == *proj, proj)
                                    .clicked()
                                {
                                    request_info(RequestInfoType::SelectProject(proj.clone()));
                                }
                            }
                        });
                } else if !self.oauth_token.is_empty() {
                    ui.label("No projects found or loading...");
                }
            }
        }

        ui.separator();

        ui.heading("Model");
        ui.label("Default model for new chats");
        ui.horizontal(|ui| {
            ui.add(toggle(&mut self.inherit_chat_picker));
            help(ui, "Inherit model changes from chats", |ui| {
                ui.label("Inherit from chats");
            });
        });
        ui.add_space(2.0);
        self.model_picker.show(ui, request_info);

        ui.separator();
        ui.heading("Behavior");
        ui.horizontal(|ui| {
            ui.add(toggle(&mut self.use_streaming));
            help(ui, "Receive the response as it's being generated. Disabling this will wait for the full response before displaying it", |ui| {
                ui.label("Stream response");
            });
        });
        ui.horizontal(|ui| {
            ui.add(toggle(&mut self.include_thoughts_in_history));
            help(ui, "When enabled, the model's 'thought' parts are appended to the session context for subsequent requests. Warning: This will rapidly increase token consumption", |ui| {
                ui.label("Persist Thoughts in Context");
            });
        });
        ui.horizontal(|ui| {
            ui.add(toggle(&mut self.public_file_upload));
            help(ui, "When enabled, files will be uploaded to Google's servers (File API) instead of being sent as base64 inline data. Uploaded files are temporary and only accessible by you.", |ui| {
                ui.label("Upload files (File API)");
            });
        });

        ui.separator();

        ui.heading("Miscellaneous");

        let mut enabled = self.proxy_path.is_some();
        ui.horizontal(|ui| {
            ui.add(toggle(&mut enabled));
            help(ui, "Use the proxy for gemini api request", |ui| {
                ui.label("Use proxy");
            });
        });
        if !enabled {
            self.proxy_path = None;
        } else if self.proxy_path.is_none() {
            self.proxy_path = Some(String::from("socks5://127.0.0.1:2080"));
        }

        if let Some(ref mut template) = self.proxy_path {
            ui.add(
                egui::TextEdit::singleline(template).hint_text("http://your_proxy_address:port"),
            );
        }

        // ui.toggle_value(&mut self.let_it_snow, "Let It Snow!");
        if ui.add(egui::Button::new("Let It Snow!").selected(self.let_it_snow)).clicked() {
            self.let_it_snow = !self.let_it_snow;
        }

        let mut zoom = ui.ctx().zoom_factor();

        ui.horizontal(|ui| {
            zoom_control_widget(ui, &mut zoom);
            help(ui, "Adjust the overall size of the user interface", |ui| {
                ui.label("UI Scale");
            });
        });

        if (ui.ctx().zoom_factor() - zoom).abs() > 0.001 {
            ui.ctx().set_zoom_factor(zoom);
        }

        ui.label("Reset global settings to defaults");
        if ui.button("Reset").clicked() {
            modal.open();
        }

        ui.label("Save and load settings as JSON");
        ui.horizontal(|ui| {
            if ui.button("Save").clicked() {
                let settings = self.clone();
                tokio::spawn(async move {
                    Self::ask_save_settings(settings).await;
                });
            }
            if ui.button("Load").clicked() {
                request_info(RequestInfoType::LoadSettings);
            }
        });
    }
}

#[cfg(feature = "tts")]
pub(crate) fn sanitize_text_for_tts(s: &str) -> String {
    let mut result = String::new();
    let mut start = 0;
    result.push_str(&s[start..]);
    result
}

fn zoom_control_widget(
    ui: &mut egui::Ui,
    current_zoom: &mut f32
) {
    let height = 20.0;
    let btn_width = 24.0;
    let rounding = 4.0;

    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 0.0;

        let btn_minus = egui::Button::new("-")
            .min_size(egui::vec2(btn_width, height))
            .corner_radius(egui::CornerRadius {
                nw: rounding as u8, sw: rounding as u8, ne: 0, se: 0
            });

        if ui.add(btn_minus).clicked() {
            *current_zoom = (*current_zoom - 0.1).max(0.2);
        }

        let text_area_size = egui::vec2(45.0, height);
        let (rect, _resp) = ui.allocate_exact_size(text_area_size, egui::Sense::hover());

        if ui.is_rect_visible(rect) {
            ui.painter().rect(
                rect,
                0.0,
                ui.visuals().extreme_bg_color,
                egui::Stroke::NONE,
                egui::StrokeKind::Inside
            );

            ui.painter().text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                format!("{:.0}%", *current_zoom * 100.0),
                egui::FontId::monospace(12.0),
                ui.visuals().text_color(),
            );
        }

        let btn_plus = egui::Button::new("+")
            .min_size(egui::vec2(btn_width, height))
            .corner_radius(egui::CornerRadius {
                nw: 0, sw: 0, ne: rounding as u8, se: rounding as u8
            });

        if ui.add(btn_plus).clicked() {
            *current_zoom = (*current_zoom + 0.1).min(2.0);
        }
    });
}

pub fn thinking_icon(
    ui: &mut egui::Ui,
    openness: f32,
    response: &egui::Response,
    done_thinking: bool,
) {
    let color = ui
        .style()
        .interact(response)
        .fg_stroke
        .color
        .gamma_multiply(openness.max(0.4));
    let rect = response.rect;
    let center = rect.center();

    let grid_size = 4.0;
    let spacing = rect.height() / (grid_size + 1.0) + openness / 4.0;

    for x in 0..4 {
        for y in 0..4 {
            let offset_x = (x as f32 - 1.0) * spacing;
            let offset_y = (y as f32 - 1.5) * spacing;
            let pos = center + egui::vec2(offset_x, offset_y) * 2.2;

            let distance_to_center = (x as f32 - 1.5).hypot(y as f32 - 1.5);
            let radius = egui::lerp(1.0..=2.0, 1.0 - distance_to_center);

            if radius > 0.1 {
                if !done_thinking {
                    let anim = (ui.input(|i| i.time) + (x as f64 + y as f64) / 16.0) % 0.9;
                    ui.painter().circle_filled(pos, radius + anim as f32, color);
                } else {
                    ui.painter().circle_filled(pos, radius, color);
                }
            }
        }
    }
}
