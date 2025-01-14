import click
import json
import os
from src._core import logging
from src.resume.utils import (
    generate_resume_for_job,
    setup_resume_llm,
    md_to_html,
    list_to_md,
    save_pdf,
)
from src.resume.schema import ResumeBase


"""
This script generates a PDF resume based on a JSON resume, a Markdown template, and a job description.
Functions:
    generate_resume(resume_json, template_md, job_description, output_pdf):
        Command-line interface function to generate a PDF resume.
Usage:
    Run this script from the command line with the appropriate options:
    python generate.py --resume-json <path_to_resume_json> --template-md <path_to_template_md> --job-description <job_description> --output-pdf <path_to_output_pdf>
Options:
    --resume-json: Path to the JSON file containing the resume data. Default is "resume.json".
    --template-md: Path to the Markdown template file. Default is "src/resume/template.md".
    --job-description: Job description for the position. This option is required.
    --output-pdf: Path to save the generated PDF resume. Default is "output.pdf".
Example:
    python generate.py --resume-json resume.json --template-md src/resume/template.md --job-description "Software Engineer at XYZ Corp" --output-pdf output.pdf
"""


@click.command()
@click.option(
    "--resume-json",
    default="resume.json",
    help="Path to the JSON file containing the resume data.",
)
@click.option(
    "--template-md",
    default=os.path.join("src", "resume", "template.md"),
    help="Path to the Markdown template file.",
)
@click.option(
    "--job-description", required=True, help="Job description for the position."
)
@click.option(
    "--output-pdf", default="output.pdf", help="Path to save the generated PDF resume."
)
def generate_resume(resume_json, template_md, job_description, output_pdf):
    """Generate a PDF resume based on a JSON resume, template, and job description."""
    # Load resume data
    logging.info(f"Loading resume data from {resume_json}")
    with open(resume_json, "r", encoding="utf-8") as json_file:
        resume = ResumeBase(**json.load(json_file))

    # Load template
    logging.info(f"Loading template from {template_md}")
    with open(template_md, "r", encoding="utf-8") as file:
        template = file.read()

    # Setup and generate resume content
    logging.info("Setting up resume LLM and generating resume content")
    llm = setup_resume_llm(resume)

    logging.info(f"Generating resume for job: {job_description[:50]}")
    generated_resume = generate_resume_for_job(llm, job_description, resume)

    bio = generated_resume.bio
    title = generated_resume.title
    experiences = list_to_md(generated_resume.experiences)
    educations = list_to_md(generated_resume.educations)
    courses = list_to_md(generated_resume.courses)
    publications = list_to_md(generated_resume.publications)
    projects = list_to_md(generated_resume.projects)
    skills = list_to_md(generated_resume.skills)

    filled = template.format(
        bio=bio,
        skills=skills,
        projects=projects,
        publications=publications,
        courses=courses,
        experiences=experiences,
        title=title,
        educations=educations,
    )

    # Convert to HTML
    logging.info("Converting Markdown to HTML")
    html = md_to_html(filled)

    # Save to PDF
    logging.info(f"Saving generated resume to {output_pdf}")
    save_pdf(html, output_pdf)

    logging.info("Resume generation complete")


if __name__ == "__main__":
    generate_resume()
