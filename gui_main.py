import dearpygui.dearpygui as dpg
import os.path
import sys
import re
from pathlib import Path
from time import strftime, gmtime
from subprocess import PIPE, Popen, STDOUT
from datetime import datetime
from utils import save_config_ini, get_config_ini, get_template_file, templates_list, templates_update, get_img_ratio, resize_img

if __name__ == "__main__":

    def load_config(section = "DEFAULT", ini_file=True, custom_values = []):
        values = get_config_ini(section) if ini_file else custom_values
        if "model_folder" in values: dpg.set_value("t_txt_model_folder", values["model_folder"]),
        if "output_folder" in values: dpg.set_value("t_txt_output_folder", values["output_folder"]),
        if "prompt" in values: dpg.set_value("t_prompt", values["prompt"]),
        if "width" in values: dpg.set_value("t_width", int(values["width"])),
        if "height" in values: dpg.set_value("t_height", int(values["height"])),
        if "iter" in values: dpg.set_value("t_iter", int(values["iter"])),
        if "samples" in values: dpg.set_value("t_samples", int(values["samples"])),
        if "steps" in values: dpg.set_value("t_steps", int(values["steps"])),
        if "scale" in values: dpg.set_value("t_scale", float(values["scale"])),
        if "ddim_eta" in values: dpg.set_value("t_ddim_eta", float(values["ddim_eta"])),
        if "skip_save" in values: dpg.set_value("t_skip_save", values["skip_save"] == "True"),
        if "skip_grid" in values: dpg.set_value("t_skip_grid", values["skip_grid"] == "True"),
        if "grid_rows" in values: dpg.configure_item("t_n_rows", show=values["skip_grid"] == "False")
        if "grid_rows" in values: dpg.set_value("t_n_rows", int(values["grid_rows"])),
        if "optimized_mode" in values: dpg.set_value("t_optimized_mode", values["optimized_mode"] == "True"),
        if "optimized_mode" in values: dpg.configure_item("t_group_optimized", show=values["optimized_mode"] == "True")
        if "turbo" in values: dpg.set_value("t_turbo", values["turbo"] == "True"),
        if "unet_bs" in values: dpg.set_value("t_unet_bs", int(values["unet_bs"])),
        if "format" in values: dpg.set_value("t_format", values["format"]),
        if "device" in values: dpg.set_value("t_device", values["device"]),
        if "seed" in values: dpg.set_value("t_seed", int(values["seed"])),
        if "refimg_path" in values: 
            dpg.set_value("t_refimg_txt", values["refimg_path"])
            if values["refimg_path"] != "":
                set_ref_img(values["refimg_path"])
                dpg.set_value("t_strength", float(values["strength"]))
            else:
                if dpg.does_item_exist("t_refimg_img"):
                    cb_refimg_clear("", None)


    def save_config(section = "LAST", ini_file=True):
        values = {
            "model_folder": dpg.get_value("t_txt_model_folder"),
            "output_folder": dpg.get_value("t_txt_output_folder"),
            "prompt": dpg.get_value("t_prompt").strip().replace("\n", " "),
            "width": str(dpg.get_value("t_width")),
            "height": str(dpg.get_value("t_height")),
            "iter": str(dpg.get_value("t_iter")),
            "samples": str(dpg.get_value("t_samples")),
            "steps": str(dpg.get_value("t_steps")),
            "scale": str(dpg.get_value("t_scale")),
            "ddim_eta": str(dpg.get_value("t_ddim_eta")),
            "skip_save": str(dpg.get_value("t_skip_save")),
            "skip_grid": str(dpg.get_value("t_skip_grid")),
            "grid_rows": str(dpg.get_value("t_n_rows")),
            "precision": dpg.get_value("t_precision"),
            "optimized_mode": str(dpg.get_value("t_optimized_mode")),
            "turbo": str(dpg.get_value("t_turbo")),
            "unet_bs": str(dpg.get_value("t_unet_bs")),
            "format": dpg.get_value("t_format"),
            "device": dpg.get_value("t_device"),
            "seed": str(dpg.get_value("t_seed")),
            "refimg_path": dpg.get_value("t_refimg_txt"),
            "strength": str(dpg.get_value("t_strength"))
        }
        if ini_file:
            save_config_ini(values, section)
        return values


    dpg.create_context()
    dpg.create_viewport(title="Stable Diffusion", width=1768, height=992, x_pos=300, y_pos=200)
    dpg.setup_dearpygui()
    dpg.set_global_font_scale(1.3)

    ini_config = get_config_ini()
    optimized_installed = os.path.isfile('./optimizedSD/optimized_img2img.py') and os.path.isfile('./optimizedSD/optimized_txt2img.py')

    def cb_generate(sender, app_data):

        # Save last config into ini
        values = save_config()

        prompt_clean = dpg.get_value("t_prompt").strip().replace("\n", " ")
        arg_list = []
        arg_list.append("--prompt")
        arg_list.append(prompt_clean)

        img2img_mode = dpg.does_item_exist("t_refimg_img")
        if img2img_mode:
            arg_list.append("--strength")
            arg_list.append(values["strength"])
            arg_list.append("--init-img")
            arg_list.append(values["refimg_path"])
            
        arg_list.append("--W")
        arg_list.append(values["width"])
        arg_list.append("--H")
        arg_list.append(values["height"])

        if values["optimized_mode"] == "True" and optimized_installed:
            arg_list.append("--format")
            arg_list.append(values["format"])
            if values["turbo"] == "True":
                arg_list.append("--turbo")
            arg_list.append("--unet_bs")
            arg_list.append(values["unet_bs"])
            arg_list.append("--device")
            arg_list.append(values["device"])
        if values["model_folder"] != "True" and os.path.isfile(values["model_folder"]):
            arg_list.append("--ckpt")
            arg_list.append(values["model_folder"])
        if values["skip_grid"] == "True":
            arg_list.append("--skip_grid")
        else:
            if values["grid_rows"] != 0 and (values["grid_rows"] != values["samples"]):
                arg_list.append("--n_rows")
                arg_list.append(values["grid_rows"])
        if values["skip_save"] == "True":
            arg_list.append("--skip_save")
        if values["output_folder"] != "" and os.path.isdir(values["output_folder"]):
            arg_list.append("--outdir")
            arg_list.append(values["output_folder"])
        if values["seed"] != "-1":
            arg_list.append("--seed")
            arg_list.append(values["seed"])
        arg_list.append("--precision")
        arg_list.append(values["precision"])
        arg_list.append("--n_iter")
        arg_list.append(values["iter"])
        arg_list.append("--n_samples")
        arg_list.append(values["samples"])
        arg_list.append("--ddim_steps")
        arg_list.append(values["steps"])
        arg_list.append("--ddim_eta")
        arg_list.append(values["ddim_eta"])
        arg_list.append("--scale")
        arg_list.append(values["scale"])

        # print("===== arg_list -> ", arg_list)
        dpg.configure_item("t_generate", enabled=False)

        delete_images()

        dpg.show_item("t_gen_modal")
        dpg.show_item("t_gen_modal_loadwheel")

        try:
            if values["optimized_mode"] == "True":
                script_to_execute = './optimizedSD/optimized_img2img.py' if img2img_mode else './optimizedSD/optimized_txt2img.py'
            else:
                script_to_execute = './scripts/img2img.py' if img2img_mode else './scripts/txt2img.py'

            sd_process = Popen(
                [sys.executable, script_to_execute, *arg_list],
                stdout=PIPE,
                stderr=STDOUT,
                shell=True,
                encoding='utf-8',
                errors='replace'
            )

            output_folder = values["output_folder"] if values["output_folder"] != "" and os.path.isdir(values["output_folder"]) else ""
            while True:
                line = sd_process.stdout.readline().strip()
                # print(line, flush=True)
                if line != "":
                    dpg.set_value("t_gen_modal_txt", line)
                percent = re.search("\d+(?:\.\d+)?%", line)
                if percent is not None:
                    dpg.set_value("t_generate_progress", float(percent.group(0)[:-1])/100)
                    dpg.set_value("t_generate_progress_txt", str(int(percent.group(0)[:-1])) + "%")

                if output_folder == "": # if output_folder is empty or not valid, look for the output folder if optimized code is used
                    output_idx = line.find("exported to")
                    if output_idx != -1:
                        output_folder = line[output_idx+12:].strip()

                if line == '' and sd_process.poll() != None:
                    break


            if output_folder == "": # output_folder is still empty, set default output folder
                output_folder = "outputs/img2img-samples/samples" if img2img_mode else "outputs/txt2img-samples/samples" 

            if os.path.isdir(output_folder):
                result_folder = os.getcwd() + "/" + output_folder
                cb_img_preview_dialog(sender="", app_data = {"file_path_name": result_folder})
                # TODO option to choose to open the corresponding folder or not
                # os.startfile(result_folder)

        except Exception as e:
            dpg.set_value("t_gen_modal_txt", e)

        dpg.hide_item("t_gen_modal_loadwheel")
        dpg.enable_item("t_gen_modal_ok_btn")
        dpg.configure_item("t_gen_modal_ok_btn", label="OK")
        dpg.enable_item("t_generate")


    def cb_refimg_dialog(sender, app_data):
        set_ref_img((app_data["file_path_name"]))


    def set_ref_img(path):
        if os.path.isfile(path) and path.lower().endswith((".png", ".jpg")):
            if dpg.does_item_exist("t_refimg_img"):
                dpg.delete_item("t_refimg_img")
            dpg.show_item("t_group_refimg")
            dpg.set_value("t_refimg_txt", path)
            image, size = resize_img(path, (150, 150))
            with dpg.texture_registry() as reg_id:
                texture_id = dpg.add_static_texture(size[0], size[1], image, parent=reg_id)
            dpg.add_image(texture_id, tag="t_refimg_img", user_data=path, before="t_strength")


    def cb_refimg_clear(sender, app_data):
        dpg.hide_item("t_group_refimg")
        dpg.set_value("t_refimg_txt", "")
        if dpg.does_item_exist("t_refimg_img"):
            dpg.delete_item("t_refimg_img")


    def cb_refimg_txt_changed(sender, app_data):
        try:
            if os.path.isfile(app_data) and app_data.lower().endswith((".png", ".jpg")):
                set_ref_img(app_data)
        except: pass


    def cb_model_dialog(sender, app_data):
        dpg.set_value("t_txt_model_folder", app_data["file_path_name"])
        dpg.show_item("t_model_clear_btn")


    def cb_output_dialog(sender, app_data):
        # app_data["file_path_name"] app_data["file_name"]
        dpg.set_value("t_txt_output_folder", app_data["file_path_name"])
        dpg.show_item("t_output_clear_btn")


    def cb_size_changed():
        curr_width = dpg.get_value("t_width")
        curr_height = dpg.get_value("t_height")
        ratio = get_img_ratio(curr_width, curr_height)
        dpg.set_value("t_aspect_txt", "Aspect ratio: " + str(ratio[0]) + ":" + str(ratio[1]))


    # File/folder browse items
    with dpg.file_dialog(directory_selector=False, width=800, height=600, show=False, callback=cb_refimg_dialog, tag="t_refimg_file_dialog"):
        dpg.add_file_extension(".png")
        dpg.add_file_extension(".jpg")
        # dpg.add_file_extension(".*")
    with dpg.file_dialog(directory_selector=False, width=800, height=600, show=False, callback=cb_model_dialog, tag="t_model_file_dialog"):
        dpg.add_file_extension(".ckpt")
        

    dpg.add_file_dialog(label="Output Folder", width=800, height=600,  directory_selector=True, show=False, callback=cb_output_dialog, tag="t_output_file_dialog")

    with dpg.window(label="Main", tag="primary"):
        dpg.add_text("Load ini configs / Load or save templates")
        with dpg.group(horizontal=True, horizontal_spacing=10):
            dpg.add_button(label="Config Default", tag="t_config_default", width=150, height=30, callback=lambda: load_config("DEFAULT"))
            with dpg.tooltip("t_config_default"):
                dpg.add_text("Loads the default configuration")
            dpg.add_button(label="Config Last", tag="t_config_last",  width=150, height=30, callback=lambda: load_config("LAST"))
            with dpg.tooltip("t_config_last"):
                dpg.add_text("Loads the configuration from the last generation (Saved when the Generate button is pressed")
            dpg.add_text("|")
            dpg.add_button(label="Load template", tag="t_templates_load_btn", width=150, height=30, callback=lambda: (dpg.show_item("t_templates_win_modal"), create_list_templates()))
            with dpg.tooltip("t_templates_load_btn"):
                dpg.add_text("Opens a list of saved templates from the templates.csv file")
            dpg.add_button(label="Save template", tag="t_templates_save_btn", width=150, height=30)
            with dpg.popup(dpg.last_item(), mousebutton=dpg.mvMouseButton_Left, modal=True, tag="t_template_saved_modal"):
                dpg.add_text("Template saved succefully")
                dpg.add_button(label="Ok", width=80, height=30, callback=lambda: (dpg.hide_item("t_template_saved_modal"), cb_save_template()))
            with dpg.tooltip("t_templates_save_btn"):
                dpg.add_text("Saves the current template (prompt and current config) into the templates.csv file")

        dpg.add_spacer(height=10)
        dpg.add_separator()

        with dpg.group(horizontal=True, horizontal_spacing=10):
            dpg.add_button(label="Model Folder", width=130, tag="t_select_model_folder", callback=lambda: dpg.show_item("t_model_file_dialog"))
            with dpg.tooltip("t_select_model_folder"):
                dpg.add_text("Path to checkpoint of model, if empty, uses 'models/ldm/stable-diffusion-v1/model.ckpt'")
            dpg.add_input_text(label="", width=dpg.get_viewport_width()/3, tag="t_txt_model_folder", default_value=ini_config["model_folder"], callback=lambda sender, app_data: dpg.show_item("t_model_clear_btn") if app_data != "" else dpg.hide_item("t_model_clear_btn"))
            dpg.add_button(label="Clear", tag="t_model_clear_btn", show=(ini_config["model_folder"] != ""), callback=lambda: (dpg.set_value("t_txt_model_folder", ""), dpg.hide_item("t_model_clear_btn")))

        dpg.add_spacer(height=10)

        with dpg.group(horizontal=True, horizontal_spacing=10):
            dpg.add_button(label="Output Folder", width=130, tag="t_select_output_folder", callback=lambda: dpg.show_item("t_output_file_dialog"))
            with dpg.tooltip("t_select_output_folder"):
                dpg.add_text("The folder to place the generated images in, if empty or not valid, the default Stable Diffusion folder will be used (project's root \"outputs\" folder)")
            dpg.add_input_text(label="", width=dpg.get_viewport_width()/3, tag="t_txt_output_folder", default_value=ini_config["output_folder"], callback=lambda sender, app_data: dpg.show_item("t_output_clear_btn") if app_data != "" else dpg.hide_item("t_output_clear_btn"))
            dpg.add_button(label="Clear", tag="t_output_clear_btn", show=(ini_config["output_folder"] != ""), callback=lambda: (dpg.set_value("t_txt_output_folder", ""), dpg.hide_item("t_output_clear_btn")))

        dpg.add_spacer(height=10)

        with dpg.group(horizontal=True, horizontal_spacing=10):
            dpg.add_button(label="Reference Image", tag="t_select_refimg", callback=lambda: dpg.show_item("t_refimg_file_dialog"))
            with dpg.tooltip("t_select_refimg"):
                dpg.add_text("Select the path to an image, if an image is set here the image to image mode of Stable Diffusion will be used")
            dpg.add_input_text(width=480, tag="t_refimg_txt", default_value=ini_config["refimg_path"], callback=cb_refimg_txt_changed)
        with dpg.group(tag="t_group_refimg", show=ini_config["refimg_path"] != ""):
            dpg.add_slider_double(label="Strength", width=200, min_value=0.1, max_value=1.0, tag="t_strength", format="%.1f", default_value=float(ini_config["strength"]))
            with dpg.tooltip("t_strength"):
                dpg.add_text("Prompt strength when using reference image, for noising/unnoising. 1.0 corresponds to full destruction of information in init image")
            dpg.add_button(label="Clear", tag="t_refimg_clear_btn", callback=cb_refimg_clear)

        dpg.add_spacer(height=10)

        # Select file
        # "--from-file",
        # type=str,
        # help="if specified, load prompts from this file",

        dpg.add_input_text(label="Prompt", width=500, tag="t_prompt", multiline=True, default_value=ini_config["prompt"])
        with dpg.tooltip("t_prompt"):
            dpg.add_text("The prompt to render")

        dpg.add_spacer(height=10)

        dpg.add_text(tag="t_aspect_txt")
        dpg.add_input_int(label="Width", width=150, tag="t_width", step=32, min_value=32, max_value=4096, min_clamped=True, max_clamped=True, default_value=int(ini_config["width"]), callback=cb_size_changed)
        dpg.add_input_int(label="Height", width=150, tag="t_height", step=32, min_value=32, max_value=4096, min_clamped=True, max_clamped=True, default_value=int(ini_config["height"]), callback=cb_size_changed)
        with dpg.tooltip("t_width"):
            dpg.add_text("Image width, in pixel space, step 32")
        with dpg.tooltip("t_height"):
            dpg.add_text("Image height, in pixel space, step 32")

        dpg.add_spacer(height=10)

        dpg.add_input_int(label="Iterations", width=150, tag="t_iter", step=1, default_value=int(ini_config["iter"]))
        with dpg.tooltip("t_iter"):
            dpg.add_text("Sample this often")

        dpg.add_spacer(height=10)

        dpg.add_input_int(label="Samples", width=150, tag="t_samples", step=1, default_value=int(ini_config["samples"]))
        with dpg.tooltip("t_samples"):
            dpg.add_text("How many samples to produce for each given prompt. A.k.a. batch size")

        dpg.add_spacer(height=10)

        dpg.add_slider_int(label="Steps", width=200, min_value=1, max_value=100, tag="t_steps", default_value=int(ini_config["steps"]))
        with dpg.tooltip("t_steps"):
            dpg.add_text("Number of ddim sampling steps")

        dpg.add_spacer(height=10)

        dpg.add_slider_double(label="Scale", width=200, min_value=1.0, max_value=100.0, tag="t_scale", format="%.1f", no_input=True, default_value=float(ini_config["scale"]))
        with dpg.tooltip("t_scale"):
            dpg.add_text("Unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))")

        dpg.add_spacer(height=10)

        dpg.add_slider_double(label="Ddim eta", width=200, min_value=1.0, max_value=100.0, tag="t_ddim_eta", format="%.1f", no_input=True, default_value=float(ini_config["ddim_eta"]))
        with dpg.tooltip("t_ddim_eta"):
            dpg.add_text("Ddim eta (eta=0.0 corresponds to deterministic sampling")

        dpg.add_spacer(height=10)

        dpg.add_checkbox(label="Skip save", tag="t_skip_save", default_value=ini_config["skip_save"] == "True")
        with dpg.tooltip("t_skip_save"):
            dpg.add_text("Do not save individual samples. For speed measurements")

        with dpg.group(horizontal=True, horizontal_spacing=20):
            dpg.add_checkbox(label="Skip grid", tag="t_skip_grid", callback=lambda sender, app_data: dpg.configure_item("t_n_rows", show=(not app_data)), default_value=ini_config["skip_grid"] == "True")
            with dpg.tooltip("t_skip_grid"):
                dpg.add_text("Do not save a grid, only individual samples. Helpful when evaluating lots of samples")
            dpg.add_input_int(label="Grid rows", width=150, tag="t_n_rows", step=1, show=ini_config["skip_grid"] == "False", default_value=int(ini_config["grid_rows"]))
            with dpg.tooltip("t_n_rows"):
                dpg.add_text("Rows in the grid (default: n_samples)")

        dpg.add_spacer(height=10)

        dpg.add_combo(label="Precision", tag="t_precision", width=150, items=["autocast", "full"], default_value=ini_config["precision"])
        with dpg.tooltip("t_precision"):
            dpg.add_text("Evaluate at this precision")

        dpg.add_spacer(height=10)
        dpg.add_separator()

        dpg.add_checkbox(label="Optimized Mode", tag="t_optimized_mode", enabled=optimized_installed, default_value=optimized_installed and ini_config["optimized_mode"] == "True", callback=lambda: dpg.configure_item("t_group_optimized", show=optimized_installed and dpg.get_value("t_optimized_mode")))
        with dpg.tooltip("t_optimized_mode"):
            dpg.add_text("Uses optimized version scripts if available, required: github.com/basujindal/stable-diffusion")

        # Optimized only options
        with dpg.group(tag="t_group_optimized", show=optimized_installed and ini_config["optimized_mode"] == "True"):

            dpg.add_spacer(height=10)

            # with dpg.group(horizontal=True, horizontal_spacing=10):
            dpg.add_checkbox(label="Turbo", tag="t_turbo", default_value=ini_config["turbo"] == "True")
            with dpg.tooltip("t_turbo"):
                dpg.add_text("Reduces inference time on the expense of 1GB VRAM")

            dpg.add_input_int(label="Unet bs", width=150, tag="t_unet_bs", default_value=int(ini_config["unet_bs"]))
            with dpg.tooltip("t_unet_bs"):
                dpg.add_text("Slightly reduces inference time at the expense of high VRAM (value > 1 not recommended )")

            dpg.add_combo(label="Format", tag="t_format", width=150, items=["png", "jpg"], default_value=ini_config["format"])
            with dpg.tooltip("t_format"):
                dpg.add_text("Output image format")

            dpg.add_input_text(label="Device", width=150, tag="t_device", default_value=ini_config["device"])
            with dpg.tooltip("t_device"):
                dpg.add_text("Specify GPU (cuda/cuda:0/cuda:1/...), Default: cuda")

            dpg.add_separator()

        dpg.add_spacer(height=10)

        with dpg.group(horizontal=True, horizontal_spacing=10):
            dpg.add_input_int(label="Seed", width=150, tag="t_seed", step=0, default_value=int(ini_config["seed"]))
            with dpg.tooltip("t_seed"):
                dpg.add_text("The seed (for reproducible sampling)")
            dpg.add_button(label="Reset", callback=lambda: dpg.set_value("t_seed", -1))

        dpg.add_spacer(height=10)

        dpg.add_button(label="Generate", tag="t_generate", width=150, height=50, callback=cb_generate)
        with dpg.tooltip("t_generate"):
            dpg.add_text("Generate image(s) with the above options to the output folder")


    def modal_ok_button(sender, app_data):
        dpg.configure_item("t_gen_modal", show=False)
        dpg.disable_item("t_gen_modal_ok_btn")
        dpg.configure_item("t_gen_modal_ok_btn", label="Generating...")
        dpg.hide_item("t_gen_modal_loadwheel"), dpg.set_value("t_gen_modal_txt", "")
        dpg.set_value("t_generate_progress", 0)
        dpg.set_value("t_generate_progress_txt", "0%")


    with dpg.window(label="Generating", width=400, pos=(dpg.get_viewport_width()/2-190, dpg.get_viewport_height()/2-150), no_move=True, no_resize=True, modal=True, show=False, tag="t_gen_modal", no_title_bar=True):
        dpg.add_text("\n\n\n", tag="t_gen_modal_txt", wrap=380)
        dpg.add_separator()
        with dpg.group(horizontal=True, horizontal_spacing=10):
            dpg.add_loading_indicator(label="Test button toggle", tag="t_gen_modal_loadwheel", show=False)
            dpg.add_progress_bar(label="Progress", tag="t_generate_progress", width=-1, height=50)
        with dpg.group(horizontal=True, horizontal_spacing=20):
            dpg.add_button(label="Generating...", width=150, height=50, tag="t_gen_modal_ok_btn", enabled=False, callback=modal_ok_button)
            dpg.add_text(label="0%", tag="t_generate_progress_txt")

        # Normal mode options
        # "--plms",
        # action='store_true',
        # help="use plms sampling",

    # ============= Folder Preview window =============

    def set_preview_img(img_user_data):
        dpg.configure_item("t_txt_img_preview", user_data=img_user_data)
        if os.path.isfile(img_user_data["path"]):
            dpg.set_value("t_txt_img_preview", "--- File name:\n" + img_user_data["file_name"] + "\n--- Created:\n" + img_user_data["created"] + "\n--- Path:\n" + img_user_data["path"] + "\n--- Seed:\n" + img_user_data["seed"])
            dpg.hide_item("t_img_preview_info")
            dpg.show_item("t_group_selected_img")

            width, height, channels, data = dpg.load_image(img_user_data["path"])
            with dpg.texture_registry() as reg_id:
                texture_id = dpg.add_static_texture(width, height, data, parent=reg_id)

            if dpg.does_item_exist("t_slected_preview_img"): # Replace image if item already exists
                dpg.configure_item("t_slected_preview_img", texture_tag=texture_id, width=width, height=height)
            else:
                dpg.add_image(texture_id, tag="t_slected_preview_img", before="t_txt_img_preview")


    def cb_select_as_ref():
        cb_refimg_dialog("", app_data={"file_path_name":dpg.get_item_user_data("t_txt_img_preview")["path"], 
                                        "file_name":dpg.get_item_user_data("t_txt_img_preview")["file_name"]})


    def cb_preview_img_button(sender, app_data):
        img_user_data = dpg.get_item_user_data(sender)
        set_preview_img(img_user_data)
        
        
    def delete_images():
        dpg.delete_item("t_thunbnail_group", children_only=True)
        dpg.show_item("t_img_preview_info")
        dpg.set_value("t_txt_img_preview_folder", "")
        dpg.hide_item("t_img_preview_clear_btn")
        dpg.hide_item("t_group_selected_img")
        dpg.enable_item("t_select_img_preview_folder")
        if dpg.does_item_exist("t_slected_preview_img"):
            dpg.delete_item("t_slected_preview_img")


    def cb_txt_img_preview_folder_changed(sender, app_data):
        try:
            if os.path.isdir(app_data) and not dpg.get_item_children("t_thunbnail_group")[1]:
                cb_img_preview_dialog(sender="", app_data = {"file_path_name": app_data})
        except: pass


    def cb_img_preview_dialog(sender, app_data):
        folder_path = app_data["file_path_name"]

        dpg.set_value("t_txt_img_preview_folder", folder_path)
        dpg.show_item("t_img_preview_clear_btn")
        dpg.disable_item("t_select_img_preview_folder")

        def generate_thumbnail(img_path, thumb_data):
            image, size = resize_img(img_path, (150, 150))
            with dpg.texture_registry() as reg_id:
                texture_id = dpg.add_static_texture(size[0], size[1], image, parent=reg_id)
            # dpg.add_image(texture_id, tag="t_refimg_img", user_data=path, before="t_strength")
            return dpg.add_image_button(texture_id, tag="t_btn_img_preview_"+str(idx), parent="t_thunbnail_group", user_data=thumb_data, callback=cb_preview_img_button)
    
        try: 
            file_list = os.listdir(folder_path)
            file_list = [os.path.join(folder_path, f) for f in file_list]
            file_list.sort(key=os.path.getctime, reverse=True)
        except: file_list = []

        for idx, file_path in enumerate(file_list):
            is_valid_image = os.path.isfile(file_path) and file_path.lower().endswith((".png", ".jpg"))
            # is_dir = os.path.isdir(file_path)
            if is_valid_image:
                basename = os.path.basename(file_path)
                created_date =  strftime("%m/%d/%Y - %H:%M:%S", gmtime(os.path.getmtime(file_path))) #datetime(os.path.getmtime(file_path)).strftime('%m/%d/%Y')
                try: seed = basename.split("_")[1] if basename.find("seed") != -1 else "No seed"
                except: seed = "No seed"
                generate_thumbnail(file_path, thumb_data={"created": created_date, "file_name": basename, "path": file_path, "seed": seed, "idx": idx})


    def cb_set_seed(sender, app_data):
        seed = dpg.get_item_user_data("t_txt_img_preview")["seed"]
        if seed != "No seed":
            dpg.set_value("t_seed", int(seed))


    dpg.add_file_dialog(label="Preview Folder", width=800, height=600,  directory_selector=True, show=False, callback=cb_img_preview_dialog, tag="t_img_preview_file_dialog")

    with dpg.window(label="Folder Preview", width=200, height=dpg.get_viewport_height()-50, pos=(dpg.get_viewport_width()/2-70, 0), no_close=True, no_collapse=True):
        dpg.add_button(label="Preview Folder", width=-1, tag="t_select_img_preview_folder", callback=lambda: dpg.show_item("t_img_preview_file_dialog"))
        with dpg.tooltip("t_select_img_preview_folder"):
            dpg.add_text("The folder to preview the images of, clear before setting another folder")
        dpg.add_input_text(label="", width=-1, tag="t_txt_img_preview_folder", callback=cb_txt_img_preview_folder_changed)
        dpg.add_button(label="Clear", width=-1, height=40, tag="t_img_preview_clear_btn", show=False, callback=lambda: (dpg.set_value("t_txt_img_preview_folder", ""), dpg.hide_item("t_img_preview_clear_btn"), delete_images()))

        with dpg.group(horizontal_spacing=10, tag="t_thunbnail_group"): pass

    # ========== Preview image window ==========

    with dpg.window(label="Image preview", width=700, height=dpg.get_viewport_height()-50, pos=(dpg.get_viewport_width()-740, 0), no_close=True, no_collapse=True):
        
        dpg.add_text("<< Select an image", tag="t_img_preview_info")

        with dpg.group(tag="t_group_selected_img", show=False):
            dpg.add_input_text(width=-1, height=250, multiline=True, readonly=True, tag="t_txt_img_preview")
            with dpg.group(horizontal=True):
                dpg.add_button(label="Select as reference", height=40, tag="t_btn_select_as_ref_img", callback=cb_select_as_ref)
                dpg.add_button(label="Enter seed", height=40, tag="t_btn_preview_img_seed", callback=cb_set_seed)
                dpg.add_button(label="Open folder", height=40, callback=lambda: os.startfile(dpg.get_value("t_txt_img_preview_folder")))
    
    # ==================================================
    # ========== Templates window ==========

    template_csv_lines = []
    deleted_indexes = []

    def template_close():
        global template_csv_lines
        global deleted_indexes
        templates_update(deleted_indexes, template_csv_lines)
        deleted_indexes = []
        dpg.delete_item("t_templates_list", children_only=True)
        dpg.hide_item("t_templates_win_modal")


    def cb_apply_template(sender, app_data, user_data):
        dpg.set_value("t_prompt", user_data["prompt"])
        if "options" in user_data :
            options = user_data["options"].split(",")
            options_sep = []
            for option in options:
                options_sep.append(option.split(":"))
            opt_dict = {}
            for option in options_sep:
                opt_dict[option[0]] = option[1] 
            load_config(ini_file=False, custom_values=opt_dict)
        template_close()


    def cb_save_template():
        file = get_template_file()
        values = save_config(ini_file=False)
        values.pop("prompt")
        options = ",".join(key+":"+values[key] for key in values.keys())
        with open(file,'a+') as csv:
            csv.seek(0)
            start_of_file = csv.read(100)
            if len(start_of_file) > 0:
                lines = csv.readlines()
                if len(lines[-1]) > 0 and "\n" not in lines[-1] : csv.write("\n")
            csv.write(dpg.get_value("t_prompt").strip().replace("\n", " ") + "\t" + datetime.today().strftime("%m/%d/%Y %H:%M:%S") + "\t" + options)


    def cb_delete_template(sender, app_data, user_data):
        global deleted_indexes
        global template_csv_lines
        real_index = len(template_csv_lines)-(user_data["index"]+1)
        deleted_indexes.append(real_index)
        dpg.delete_item("t_templates_list"+str(user_data["index"]))


    def cb_template_cancel():
        template_close()

    
    def create_list_templates():
        global template_csv_lines
        template_csv_lines = templates_list()
        if template_csv_lines is not None:
            for idx, line in enumerate(reversed(template_csv_lines)):
                if line != "":
                    columns = line.split("\t")
                    if len(columns) == 3:
                        try:
                            with dpg.group(tag="t_templates_list"+str(idx), parent="t_templates_list"):
                                dpg.add_spacer(height=5)
                                dpg.add_text("Saved: " + columns[1])
                                dpg.add_spacer(height=5)
                                dpg.add_text("Prompt: " + columns[0], wrap=dpg.get_viewport_width())
                                dpg.add_spacer(height=5)
                                with dpg.group(horizontal=True, horizontal_spacing=10):
                                    dpg.add_button(label="Apply", height=30, user_data={"index": idx, "date": columns[1], "prompt": columns[0]}, callback=cb_apply_template)
                                    dpg.add_button(label="Apply with config", height=30, user_data={"index": idx, "date": columns[1], "prompt": columns[0], "options":columns[2]}, callback=cb_apply_template)
                                    dpg.add_button(label="Delete", height=30, user_data={"index": idx}, callback=cb_delete_template)
                                dpg.add_spacer(height=5)
                                dpg.add_separator()
                        except: pass


    with dpg.window(label="Templates", tag="t_templates_win_modal", width=dpg.get_viewport_width(), height=dpg.get_viewport_height(), no_move=True, no_resize=True, modal=True, show=False, no_title_bar=True):
        dpg.add_button(label="Close", height=30, callback=cb_template_cancel)
        dpg.add_separator()
        with dpg.group(tag="t_templates_list"): pass

    # ==================================================

    # Setup reference image if it exist in the last config
    cb_size_changed()
    if ini_config["refimg_path"] != "":
        set_ref_img(ini_config["refimg_path"])
        
    dpg.show_viewport()
    dpg.set_primary_window("primary", True)
    dpg.start_dearpygui()
    dpg.destroy_context()