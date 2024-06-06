# Crafty
Generate lecture videos with a single prompt!

## Installation

```bash
conda create --name crafty python=3.11
conda activate crafty
pip install -r requirements.txt
```

install MacTex or TeX Live

```bash
# e.g. on macOS or Linux
brew install --cask mactex
```

install ffmpeg

```bash
# e.g. on macOS or Linux
brew install ffmpeg
```

Once installed, you can set the ```IMAGEIO_FFMPEG_EXE``` environment variable as indicated in your script. This variable points to the FFmpeg executable, which is typically located in ```/usr/local/bin/ffmpeg``` on macOS, but the provided script suggests a Homebrew-specific path under ```/opt/homebrew/bin/ffmpeg```. Verify the correct path using:

```bash
which ffmpeg
```

Then update the environment variable accordingly in your Python script or set it in your shell profile:

```bash
export IMAGEIO_FFMPEG_EXE=$(which ffmpeg)
os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/bin/ffmpeg"
```

## Set OPENAI_API_KEY

```bash
cd "<project_dir>"
# Should replace sk-xxx to a real openai api key
echo "OPENAI_API_KEY=sk-xxx" > .env
```

## Run Native

Edit parameters in local_test.py file:

```python
para = {
    # Video generation parameters
    "if_long_videos": True,             # If true, generate long videos
    "if_short_videos": False,           # Currently do not have short videos
    "script_max_words": 100,            # Currently not used
    "slides_template_file": "3",        # Marking the template file under the folder "templates". User can put their own template file name.
    "slides_style": "simple",           # Only use it if template file is not provided
    "content_slide_pages": 30,          # Number of pages for content slides
    "if_parallel_processing": False,    # If true, use parallel processing (chapters) for video generation
    "creative_temperature": 0.5,        # Temperature for creative model

    # Course information
    "course_info": "I want to learn about the history of the United States!",
    'llm_source': 'openai',
    'temperature': 0,
    "openai_key_dir": ".env",                               # OpenAI key directory
    "results_dir": "pipeline/test_outputs/",                # Output directory
    "sections_per_chapter": 20,                             # Number of sections per chapter
    "max_note_expansion_words": 500,                        # Maximum number of words for note expansion
    "regions": ["Overview", "Examples", "Essentiality"],    # Regions for note expansion
    "if_advanced_model": False,                             # If true, use advanced model for note expansion (more expensive!)
}
```

You can specify the learning objective in

```python
"course_info": "I want to learn about the history of the United States!",
```

Run locally:

```bash
conda activate crafty
cd "<project_dir>"
python local_test.py
```

## Work flow

The project is started by running ```local_test.py```. Then ```generate_videos``` in ```dev_tasks.py``` will be called. With the steps in ```generate_videos```, we can create chapters (and sections under them) and then notes (for sections under each chapter). After getting the notes, we can take them as material to run ```VideoProcessor``` for videos generation.

### local_test.py
After running local_test.py, we call
```python
generate_videos(para)
```
imported from 
```python
pipeline.dev_tasks
```

### dev_tasks.py
Then we go through the work flow:
1. **Generate chapters:**
    - mynotes.create_chapters()
2. **Generate sections (under chapters):**
    - mynotes.create_sections()
3. **Generate note for each section (regions defined in para):**
    - mynotes.create_notes()
4. **Videos generation for each chapter:**
    - myfulllongvideos.run_parallel_processing() for processing each chapter in parallel using multiprocessing
    - myfulllongvideos.run_sequential_processing() for processing each chapter sequentially
```python
mynotes = Zeroshot_notes(para)
mynotes.create_chapters()
mynotes.create_sections()
mynotes.create_notes()
print(f"Time to create notes: {round((time.time() - st) / 60, 0)} mins of the course for the request {para['course_info'] }.")

para['course_id'] = mynotes.course_id
# print(f"\nCourse ID: {para['course_id']}")

if(para['if_long_videos']):
    myfulllongvideos = VideoProcessor(para)
    if(para['if_parallel_processing']):
        myfulllongvideos.run_parallel_processing()
    else:
        myfulllongvideos.run_sequential_processing()
```

### zeroshot_notes.py
```course_id``` defined by hashing ```self.course_info```. Output files will be saved in ```/pipeline/test_outputs/<material_type>/<course_id>/```.

For notes generation, we properly format learning topic with
```python
_extract_zero_shot_topic(self)
```
in format:
```python
{
"context": <what is the context of this course>,
"level": <what is the level of this course>,
"subject": <what is the subject of this course>,
"zero_shot_topic": <what is the zero_shot_topic of this course>
}
```
Then we go through the process of chapters generation
```python
create_chapters(self)
```
For the list of chapters, generate sections under each chapter in parallel
```python
create_sections(self)
```
The information about chapters and sections will be saved in ```chapters_and_sections.json``` under ```/pipeline/test_outputs/notes/<course_id>/```.

Next by going through each chapter, we generate notes for sections in parallel:
```python
notes_exp = robust_generate_expansions(llm, sections_list_temp, chapters_name_temp, self.course_name_textbook_chapters["Course name"], max_note_expansion_words, 3, self.regions)
```
All files saved as ```notes_set{i}.json``` under ```/pipeline/test_outputs/notes/<course_id>/```, with ```i``` is the chapter index.

Examples:
```
{
  "Introduction to the Civil Rights Movement": {
    "Overview": "The Civil Rights Movement in the United States was a pivotal period in American history that aimed to end racial segregation and discrimination against African Americans. It was a grassroots movement that gained momentum in the 1950s and 1960s, led by prominent figures such as Martin Luther King Jr., Rosa Parks, and Malcolm X. The movement sought to secure equal rights and opportunities for African Americans, including the right to vote, access to education, and an end to segregation in public spaces.",
    "Examples": "One of the most famous examples of the Civil Rights Movement is the Montgomery Bus Boycott in 1955, sparked by Rosa Parks' refusal to give up her seat to a white passenger. This boycott lasted for over a year and ultimately led to the desegregation of public buses in Montgomery, Alabama. Another significant event was the March on Washington in 1963, where Martin Luther King Jr. delivered his iconic 'I Have a Dream' speech, calling for an end to racism and for civil rights legislation to be passed. The Civil Rights Act of 1964 and the Voting Rights Act of 1965 were major legislative victories for the movement, outlawing segregation and ensuring voting rights for African Americans.",
    "Essentiality": "The Civil Rights Movement was essential in bringing about social and political change in the United States. It challenged the status quo of racial inequality and paved the way for future movements advocating for equality and justice. The sacrifices and efforts of those involved in the Civil Rights Movement have had a lasting impact on American society, inspiring generations to continue the fight for civil rights and social justice."
  },
  "The Roots of Racial Segregation": {
    "Overview": "The roots of racial segregation in the United States can be traced back to the early days of colonization when European settlers brought with them the institution of slavery. The enslavement of African Americans laid the foundation for a system of racial hierarchy that would persist for centuries. As the country expanded westward, so too did the practice of segregation, with laws and social norms enforcing the separation of races in schools, housing, and public spaces.",
    "Examples": "One of the most well-known examples of racial segregation in American history is the Jim Crow laws that were enacted in the Southern states following the Civil War. These laws mandated the separation of whites and blacks in all aspects of public life, from drinking fountains to public transportation. Another example is the Chinese Exclusion Act of 1882, which prohibited Chinese immigrants from becoming naturalized citizens and limited their ability to own property.",
    "Essentiality": "Understanding the roots of racial segregation is essential for comprehending the lasting impact it has had on American society. The legacy of segregation can still be seen today in disparities in education, wealth, and access to resources between different racial groups. By examining the historical origins of segregation, we can better understand the systemic inequalities that continue to affect marginalized communities and work towards creating a more equitable society."
  },
  "Key Figures in the Civil Rights Movement": {
    "Overview": "The Civil Rights Movement in the United States was a pivotal moment in the country's history, marked by the fight for equality and justice for African Americans. Key figures played crucial roles in leading and organizing the movement, inspiring others to join in the struggle for civil rights. These figures were instrumental in challenging segregation, discrimination, and systemic racism that plagued American society.",
    "Examples": "Some of the key figures in the Civil Rights Movement include Martin Luther King Jr., a prominent leader who advocated for nonviolent resistance and civil disobedience. Rosa Parks, known as the 'mother of the Civil Rights Movement,' sparked the Montgomery Bus Boycott by refusing to give up her seat to a white passenger. Malcolm X, a powerful orator and activist, promoted black nationalism and self-defense as a means to combat racial oppression. Other notable figures include John Lewis, a civil rights leader and congressman, and Fannie Lou Hamer, a voting rights activist and organizer.",
    "Essentiality": "These key figures in the Civil Rights Movement were essential in shaping the course of American history and bringing about significant social and political change. Their courage, determination, and leadership inspired millions of people to stand up against injustice and fight for equality. Their contributions paved the way for the passage of landmark civil rights legislation, such as the Civil Rights Act of 1964 and the Voting Rights Act of 1965, which helped dismantle segregation and secure voting rights for African Americans. The legacy of these key figures continues to resonate today, reminding us of the ongoing struggle for civil rights and the importance of standing up for justice and equality."
  },
  "Major Events and Milestones": {
    "Overview": "The Civil Rights Movement in the United States was a pivotal period in American history that aimed to end racial segregation and discrimination against African Americans. This movement, which took place primarily in the 1950s and 1960s, was characterized by nonviolent protests, sit-ins, marches, and legal challenges to discriminatory laws. Major events and milestones during this time included the Montgomery Bus Boycott, the March on Washington, the Civil Rights Act of 1964, and the Voting Rights Act of 1965.",
    "Examples": "One of the most iconic events of the Civil Rights Movement was the Montgomery Bus Boycott, sparked by Rosa Parks' refusal to give up her seat to a white passenger. This boycott lasted for over a year and ultimately led to the desegregation of public transportation in Montgomery, Alabama. Another significant event was the March on Washington in 1963, where Martin Luther King Jr. delivered his famous 'I Have a Dream' speech, calling for an end to racism and for civil and economic rights for all Americans. The Civil Rights Act of 1964 outlawed discrimination based on race, color, religion, sex, or national origin, and the Voting Rights Act of 1965 aimed to overcome legal barriers that prevented African Americans from exercising their right to vote.",
    "Essentiality": "The Civil Rights Movement was essential in bringing about significant social and political change in the United States. It paved the way for the end of legal segregation, the expansion of voting rights, and the recognition of civil rights for all Americans. The movement also inspired other marginalized groups to fight for their rights and laid the foundation for future movements for equality and justice. The events and milestones of the Civil Rights Movement continue to be remembered and celebrated as important chapters in American history."
  },
  "The Role of the Federal Government": {
    "Overview": "The role of the federal government in the Civil Rights Movement and contemporary America was crucial in shaping policies and laws that aimed to end segregation and discrimination based on race. The federal government played a significant role in enforcing civil rights legislation and protecting the rights of marginalized groups.",
    "Examples": "One key example of the federal government's role in the Civil Rights Movement is the passage of the Civil Rights Act of 1964, which outlawed discrimination based on race, color, religion, sex, or national origin. Another example is the Voting Rights Act of 1965, which aimed to overcome legal barriers at the state and local levels that prevented African Americans from exercising their right to vote. In contemporary America, the federal government continues to play a role in promoting civil rights through agencies like the Department of Justice and the Equal Employment Opportunity Commission.",
    "Essentiality": "The federal government's involvement in the Civil Rights Movement was essential in bringing about significant social change and progress towards equality for all Americans. Without federal intervention, it would have been much more difficult to dismantle the systemic racism and discrimination that existed in the United States. The federal government's continued role in protecting civil rights is essential to ensure that all individuals are treated fairly and have equal opportunities in society."
  },
  "Civil Rights Legislation": {
    "Overview": "Civil Rights Legislation refers to the laws and policies enacted by the United States government to protect the rights of individuals regardless of their race, color, religion, sex, or national origin. These laws were crucial in addressing the systemic discrimination and segregation that existed in American society, particularly against African Americans.",
    "Examples": "One of the most significant pieces of Civil Rights Legislation was the Civil Rights Act of 1964, which outlawed discrimination based on race, color, religion, sex, or national origin. This landmark legislation also ended segregation in public places and banned employment discrimination. Another important law was the Voting Rights Act of 1965, which aimed to overcome legal barriers that prevented African Americans from exercising their right to vote. Additionally, the Fair Housing Act of 1968 prohibited discrimination in the sale, rental, and financing of housing based on race, religion, national origin, or sex.",
    "Essentiality": "Civil Rights Legislation was essential in promoting equality and justice for all individuals in the United States. These laws helped to dismantle the institutionalized racism and discrimination that had been prevalent for centuries. By guaranteeing equal rights and opportunities for all citizens, Civil Rights Legislation played a crucial role in shaping a more inclusive and equitable society. It also paved the way for future generations to continue the fight for civil rights and social justice."
  },
  "The Impact of the Civil Rights Movement on Other Social Movements": {
    "Overview": "The Civil Rights Movement in the United States had a profound impact on other social movements both within the country and around the world. The fight for racial equality and justice inspired and paved the way for various marginalized groups to advocate for their rights and challenge systemic discrimination.",
    "Examples": "One significant example of the impact of the Civil Rights Movement on other social movements is the Women's Rights Movement. The activism and strategies used by civil rights leaders, such as nonviolent protests and grassroots organizing, were adopted by women fighting for gender equality. The LGBTQ+ rights movement also drew inspiration from the Civil Rights Movement, with activists using similar tactics to push for equal rights and protections. Additionally, the Disability Rights Movement and the Environmental Justice Movement were influenced by the Civil Rights Movement, as they sought to address discrimination and inequality in their respective areas.",
    "Essentiality": "Understanding the impact of the Civil Rights Movement on other social movements is essential for recognizing the interconnectedness of struggles for justice and equality. By studying how different movements have learned from and built upon the successes of the Civil Rights Movement, we can gain insights into effective strategies for creating social change and advancing human rights. Recognizing the legacy of the Civil Rights Movement in shaping contemporary social movements also highlights the ongoing work needed to address systemic inequalities and injustices in society."
  },
  "The Black Power Movement": {
    "Overview": "The Black Power Movement was a social and political movement that emerged in the 1960s as a response to the ongoing struggle for civil rights and racial equality in the United States. It emphasized racial pride, self-determination, and the need for African Americans to define and defend their own interests. The movement sought to challenge white supremacy and institutional racism through various means, including political activism, community organizing, and cultural expression.",
    "Examples": "One of the most prominent organizations associated with the Black Power Movement was the Black Panther Party, founded in 1966 in Oakland, California. The party advocated for armed self-defense against police brutality and provided community services such as free breakfast programs for children. Another key figure in the movement was Malcolm X, a civil rights activist who promoted black nationalism and self-reliance. The Black Power Movement also influenced the rise of black arts and literature, with artists and writers using their work to explore themes of black identity and empowerment.",
    "Essentiality": "The Black Power Movement was essential in shifting the focus of the civil rights movement from integration to self-determination and empowerment for African Americans. It challenged the idea that equality could only be achieved through assimilation into white society and instead emphasized the importance of embracing and celebrating black culture and heritage. The movement also highlighted the need for economic and political empowerment within black communities, leading to increased activism and advocacy for social justice issues. While the Black Power Movement faced criticism and controversy, its impact on American society and culture continues to be felt today."
  },
  "The Role of Women in the Civil Rights Movement": {
    "Overview": "The role of women in the Civil Rights Movement was crucial in shaping the movement and achieving significant progress towards racial equality in America. Women played key leadership roles, organized grassroots campaigns, and mobilized communities to fight against segregation and discrimination. They were instrumental in advocating for civil rights legislation and challenging societal norms that perpetuated inequality.",
    "Examples": "One prominent example of a woman who played a pivotal role in the Civil Rights Movement is Rosa Parks, whose refusal to give up her seat on a segregated bus in Montgomery, Alabama, sparked the Montgomery Bus Boycott. Another notable figure is Fannie Lou Hamer, a civil rights activist who co-founded the Mississippi Freedom Democratic Party and fought for voting rights for African Americans. Women like Ella Baker, Diane Nash, and Dorothy Height also made significant contributions to the movement through their organizing efforts and advocacy work.",
    "Essentiality": "The participation of women in the Civil Rights Movement was essential for its success and impact. Women brought unique perspectives, organizational skills, and determination to the movement, helping to broaden its reach and effectiveness. Their contributions challenged traditional gender roles and paved the way for greater gender equality in the broader civil rights struggle. Recognizing and honoring the role of women in the Civil Rights Movement is crucial for understanding the full scope of the movement's achievements and the ongoing fight for social justice."
  },
  "The Civil Rights Movement in Popular Culture": {
    "Overview": "The Civil Rights Movement in Popular Culture refers to the portrayal and representation of the Civil Rights Movement in various forms of media and entertainment. This includes movies, television shows, music, literature, and art that depict the struggles, triumphs, and key figures of the movement. Popular culture has played a significant role in shaping public perceptions and understanding of the Civil Rights Movement, as well as keeping its legacy alive for future generations.",
    "Examples": "Some notable examples of the Civil Rights Movement in Popular Culture include the film 'Selma' which chronicles the 1965 Selma to Montgomery voting rights marches led by Martin Luther King Jr., the television series 'Eyes on the Prize' which documents the history of the Civil Rights Movement, the song 'A Change is Gonna Come' by Sam Cooke which became an anthem for the movement, and the artwork of Jacob Lawrence who depicted scenes of the Civil Rights Movement in his paintings.",
    "Essentiality": "The representation of the Civil Rights Movement in Popular Culture is essential for several reasons. Firstly, it helps to educate and inform audiences, especially younger generations, about the history and significance of the movement. It also serves as a form of remembrance and tribute to the individuals who fought for civil rights and social justice. Additionally, popular culture can inspire activism and social change by highlighting the power of collective action and the importance of standing up against injustice."
  },
  "The Legacy of Martin Luther King Jr.": {
    "Overview": "The legacy of Martin Luther King Jr. is one of the most enduring and impactful in American history. As a prominent leader in the Civil Rights Movement, King played a crucial role in advocating for racial equality and justice. His famous 'I Have a Dream' speech, delivered during the March on Washington in 1963, continues to inspire generations of Americans to strive for a more inclusive and equitable society. King's commitment to nonviolent protest and his message of love and unity have left a lasting impression on the nation.",
    "Examples": "One of the most significant examples of Martin Luther King Jr.'s legacy is the Civil Rights Act of 1964, which outlawed discrimination based on race, color, religion, sex, or national origin. This landmark legislation was a direct result of the efforts of King and other civil rights activists who fought tirelessly for equal rights. Additionally, King's influence can be seen in the continued push for social justice and equality in contemporary America, with movements such as Black Lives Matter drawing inspiration from his teachings.",
    "Essentiality": "Understanding the legacy of Martin Luther King Jr. is essential for comprehending the progress and challenges of the Civil Rights Movement and its impact on contemporary America. King's teachings on nonviolent resistance, equality, and justice continue to resonate with individuals and communities striving for a more just and inclusive society. By recognizing and honoring King's legacy, we can continue to work towards realizing his dream of a nation where all individuals are judged not by the color of their skin, but by the content of their character."
  },
  "The Civil Rights Movement and Education": {
    "Overview": "The Civil Rights Movement had a significant impact on education in the United States. Prior to the movement, schools were segregated based on race, with African American students attending separate, often inferior, schools compared to their white counterparts. The movement sought to end this segregation and ensure equal educational opportunities for all students, regardless of race. This led to landmark Supreme Court cases such as Brown v. Board of Education in 1954, which declared segregation in public schools unconstitutional.",
    "Examples": "One of the most well-known examples of the impact of the Civil Rights Movement on education is the integration of Little Rock Central High School in Arkansas in 1957. Nine African American students, known as the Little Rock Nine, were initially prevented from entering the school by the Arkansas National Guard, but eventually attended with the help of federal troops. This event highlighted the resistance to desegregation in many parts of the country. Another example is the implementation of busing programs in cities like Boston and Charlotte to achieve racial balance in schools.",
    "Essentiality": "The Civil Rights Movement's impact on education was essential in promoting equality and opportunity for all students. By challenging the status quo of segregation in schools, the movement paved the way for greater diversity and inclusivity in educational institutions. It also brought attention to the disparities in resources and quality of education between white and minority students, leading to efforts to address these inequalities through policies like affirmative action and Title IX. The legacy of the Civil Rights Movement in education continues to shape discussions and policies around diversity, equity, and inclusion in schools today."
  },
  "The Civil Rights Movement and Economic Justice": {
    "Overview": "The Civil Rights Movement not only focused on racial equality but also on economic justice for African Americans. Many civil rights leaders believed that economic empowerment was essential for achieving true equality. They advocated for fair wages, job opportunities, and access to economic resources for African Americans who had long been marginalized and discriminated against in the workforce.",
    "Examples": "One of the key figures in the Civil Rights Movement who emphasized economic justice was Martin Luther King Jr. In his famous 'I Have a Dream' speech, King called for an end to economic disparities and for equal opportunities for all Americans. The Civil Rights Act of 1964 and the Voting Rights Act of 1965 were also important pieces of legislation that aimed to address economic injustices faced by African Americans. Additionally, organizations like the Southern Christian Leadership Conference (SCLC) and the Student Nonviolent Coordinating Committee (SNCC) worked towards economic empowerment through initiatives such as voter registration drivers and economic boycotts.",
    "Essentiality": "The fight for economic justice was essential in the Civil Rights Movement as it aimed to address the systemic inequalities that African Americans faced in the economic sphere. By advocating for fair wages, job opportunities, and access to economic resources, civil rights leaders sought to create a more equitable society where all individuals had the chance to thrive. Economic empowerment was seen as a crucial component of achieving true equality and breaking down the barriers that had long hindered African Americans from fully participating in the economic life of the nation."
  },
  "The Civil Rights Movement and Voting Rights": {
    "Overview": "The Civil Rights Movement in the United States was a pivotal time in history that aimed to end racial segregation and discrimination against African Americans. One of the key aspects of this movement was the fight for voting rights, as African Americans were often denied the right to vote through various discriminatory practices such as literacy tests, poll taxes, and intimidation tactics. The Voting Rights Act of 1965 was a landmark piece of legislation that aimed to overcome these barriers and ensure that all citizens, regardless of race, had equal access to the voting booth.",
    "Examples": "One of the most famous examples of the struggle for voting rights during the Civil Rights Movement was the march from Selma to Montgomery in 1965. Led by civil rights leaders such as Martin Luther King Jr. and John Lewis, the march aimed to draw attention to the need for federal voting rights legislation. The marchers were met with violence from state troopers on the Edmund Pettus Bridge, an event that became known as 'Bloody Sunday' and galvanized support for the Voting Rights Act.",
    "Essentiality": "The fight for voting rights during the Civil Rights Movement was essential in ensuring that all citizens had equal access to the political process. By overcoming discriminatory practices that had long prevented African Americans from voting, the Voting Rights Act of 1965 helped to enfranchise millions of citizens and pave the way for greater political representation and participation. The legacy of this struggle continues to resonate today, as efforts to protect voting rights and combat voter suppression remain ongoing challenges in American society."
  },
  "Contemporary Civil Rights Issues": {
    "Overview": "Contemporary Civil Rights Issues in the United States continue to be a significant topic of discussion and activism. These issues encompass a wide range of social, political, and economic challenges that impact marginalized communities and individuals. From systemic racism and police brutality to LGBTQ+ rights and immigration reform, the fight for civil rights in America is ongoing and multifaceted.",
    "Examples": "Some examples of contemporary civil rights issues include the Black Lives Matter movement, which seeks to address police violence and racial inequality; the fight for marriage equality for LGBTQ+ individuals; the push for comprehensive immigration reform to protect the rights of undocumented immigrants; and the ongoing struggle for gender equality and reproductive rights. These issues have sparked protests, advocacy campaigns, and legal battles across the country, highlighting the continued need for social change and justice.",
    "Essentiality": "Addressing contemporary civil rights issues is essential for creating a more just and equitable society. By advocating for the rights of marginalized communities and individuals, we can work towards dismantling systemic oppression and discrimination. It is crucial to listen to and amplify the voices of those most affected by these issues, and to actively support policies and initiatives that promote equality and justice for all. Only by confronting and addressing these challenges head-on can we truly achieve a more inclusive and fair society for all Americans."
  },
  "The Role of Technology in Modern Civil Rights Activism": {
    "Overview": "Technology has played a crucial role in modern civil rights activism, providing new tools and platforms for individuals and groups to organize, communicate, and advocate for social change. Social media platforms like Twitter, Facebook, and Instagram have been instrumental in spreading awareness, mobilizing supporters, and documenting instances of injustice in real-time. The use of hashtags, viral campaigns, and live streaming has allowed activists to reach a global audience and amplify their message like never before.",
    "Examples": "One notable example of technology's impact on civil rights activism is the Black Lives Matter movement, which gained momentum through social media campaigns and online organizing. The #BlackLivesMatter hashtag became a powerful symbol of resistance against police brutality and systemic racism. Another example is the #MeToo movement, which used social media to empower survivors of sexual harassment and assault to share their stories and hold perpetrators accountable. Additionally, online petitions, crowdfunding platforms, and virtual protests have provided new avenues for activism and fundraising.",
    "Essentiality": "Technology has become essential in modern civil rights activism by democratizing access to information, amplifying marginalized voices, and facilitating rapid response to social injustices. It has enabled activists to bypass traditional gatekeepers, connect with like-minded individuals across the globe, and create virtual communities of support. However, it is important to recognize the digital divide that exists, where not all communities have equal access to technology and the internet. As technology continues to evolve, it is crucial for activists to adapt and leverage these tools responsibly to advance the cause of social justice."
  },
  "The Black Lives Matter Movement": {
    "Overview": "The Black Lives Matter movement is a social movement that originated in the African American community to campaign against systemic racism and violence towards black people. It was founded in 2013 after the acquittal of George Zimmerman in the shooting death of Trayvon Martin. The movement gained momentum following the deaths of Michael Brown in Ferguson, Eric Garner in New York City, and Freddie Gray in Baltimore, among others. Black Lives Matter seeks to bring attention to the disproportionate violence and discrimination faced by black individuals in the United States.",
    "Examples": "One of the most notable examples of the Black Lives Matter movement's impact was the protests that erupted in cities across the country following the death of George Floyd in Minneapolis in 2020. These protests brought renewed attention to issues of police brutality and racial injustice, leading to widespread calls for police reform and racial equality. The movement has also sparked conversations about the need for systemic change in areas such as education, healthcare, and criminal justice to address the root causes of racial inequality.",
    "Essentiality": "The Black Lives Matter movement is essential in the ongoing fight for racial justice and equality in the United States. By raising awareness of the systemic racism and violence faced by black communities, the movement has pushed for meaningful change at both the local and national levels. It has inspired individuals to speak out against racial injustice, participate in protests, and advocate for policy reforms that address the underlying issues of inequality. The Black Lives Matter movement serves as a powerful reminder of the ongoing struggle for civil rights and the importance of standing up against injustice."
  },
  "The Intersection of Civil Rights and Immigration": {
    "Overview": "The intersection of civil rights and immigration in the United States has been a complex and often contentious issue throughout history. As the Civil Rights Movement gained momentum in the mid-20th century, it also brought attention to the struggles faced by immigrants, particularly those from marginalized communities. This intersection highlights the interconnectedness of various social justice movements and the importance of addressing issues of discrimination and inequality across different groups.",
    "Examples": "One notable example of the intersection of civil rights and immigration is the fight for rights and recognition of immigrant farmworkers in the United States. Organizations like the United Farm Workers, led by activists such as Cesar Chavez, worked to improve working conditions and wages for migrant workers, many of whom were immigrants from Latin America. Another example is the advocacy for the rights of undocumented immigrants, who often face discrimination and lack access to basic services due to their immigration status. The DREAM Act, which aimed to provide a pathway to citizenship for undocumented immigrants brought to the U.S. as children, is another example of efforts to address the intersection of civil rights and immigration.",
    "Essentiality": "Understanding the intersection of civil rights and immigration is essential for creating a more inclusive and equitable society. By recognizing the unique challenges faced by immigrant communities, particularly those from marginalized backgrounds, we can work towards addressing systemic inequalities and promoting social justice for all. This intersection also highlights the importance of solidarity and collaboration among different social justice movements, as issues of discrimination and oppression are often interconnected. By advocating for the rights of immigrants and addressing the root causes of inequality, we can move towards a more just and inclusive society for all."
  },
  "The Future of Civil Rights in America": {
    "Overview": "The future of civil rights in America is a topic of ongoing debate and discussion. As the country continues to evolve and face new challenges, the fight for equality and justice for all remains a crucial issue. The Civil Rights Movement of the 1960s paved the way for significant progress in areas such as desegregation, voting rights, and equal opportunity, but there is still much work to be done to ensure that all individuals are treated fairly and have access to the same rights and opportunities.",
    "Examples": "Some key areas of focus for the future of civil rights in America include addressing systemic racism and discrimination, promoting diversity and inclusion in all aspects of society, and advocating for policies that protect the rights of marginalized communities. This includes fighting for equal access to education, healthcare, housing, and employment, as well as working to dismantle barriers that prevent individuals from fully participating in society. Additionally, the future of civil rights in America will likely involve continued efforts to address issues such as police brutality, mass incarceration, and economic inequality that disproportionately impact communities of color.",
    "Essentiality": "The future of civil rights in America is essential for creating a more just and equitable society for all individuals. By continuing to push for progress in areas such as racial equality, LGBTQ rights, gender equality, and disability rights, we can work towards a future where everyone has the opportunity to thrive and succeed. It is crucial for individuals, communities, and policymakers to remain committed to the fight for civil rights and to actively work towards creating a more inclusive and equitable society for future generations."
  },
  "Conclusion: The Ongoing Struggle for Equality": {
    "Overview": "The conclusion of the Civil Rights Movement marked a significant milestone in the ongoing struggle for equality in the United States. While important legislative victories were achieved, such as the Civil Rights Act of 1964 and the Voting Rights Act of 1965, the fight for equal rights and opportunities for all Americans continues to this day. The legacy of the Civil Rights Movement serves as a reminder of the progress that has been made, but also highlights the work that still needs to be done to achieve true equality.",
    "Examples": "One example of the ongoing struggle for equality is the Black Lives Matter movement, which emerged in response to police brutality and systemic racism against Black Americans. This movement has brought attention to issues of racial injustice and inequality in the criminal justice system, sparking nationwide protests and calls for reform. Additionally, the fight for LGBTQ+ rights, gender equality, and immigrant rights are all ongoing battles for equality in contemporary America.",
    "Essentiality": "The ongoing struggle for equality is essential for creating a more just and inclusive society. By continuing to advocate for equal rights and opportunities for all individuals, we can work towards dismantling systemic barriers and discrimination that prevent certain groups from fully participating in society. It is crucial to recognize the intersectionality of different forms of oppression and to address them collectively in order to create a more equitable and fair society for all."
  }
}
```

### long_videos.py
After getting all notes, we generate videos will the following steps

1. **Create the full slides for the chapter**
2. **Generate images for the slides with only titles**
    - Currently have a dummy logic generating images for sub-title slides only.
3. **Generate scripts for each slide**
4. **Insert images into TEX file of the slides and compile PDF**
    - Based on MacTex (on Mac)
5. **Generate audio files (.mp3) for the scripts**
    - Could be improved with latest TTS progress in the future: https://bytedancespeech.github.io/seedtts_tech_report/#applications-samples
6. **Convert the full slides PDF to images**
7. **Convert the audio files to MP4 and combine them**
    - Based on ffmpeg and moviepy.editor
    - This is when your computer will start to suffer...

```python
def create_long_videos(self, chapter=0):
    """
    Create long videos for each chapter based on the notes set number.
    """
    # Create the full slides for the chapter
    self.create_full_slides(notes_set_number = chapter)  #"notes_set1"
    # Generate images for the slides with only titles
    self.create_scripts(notes_set_number = chapter)  #"notes_set1"
    # Generate scripts for each slide
    self.tex_image_generation(notes_set_number = chapter)
    # Insert images into TEX file of the slides and compile PDF
    self.insert_images_into_latex(notes_set_number = chapter)
    # Generate audio files for the scripts
    self.scripts2voice(notes_set_number = chapter)
    # Convert the full slides PDF to images
    self.pdf2image(notes_set_number = chapter)
    # Convert the audio files to MP4 and combine them
    self.mp3_to_mp4_and_combine(notes_set_number = chapter)
```

For PDF compiling:
```python
command = ['/Library/TeX/texbin/xelatex', tex_file_path]
subprocess.run(command, cwd=working_directory)
```

## Time consuming and cost

At present, the total time required to generate a script for a chapter video using GPT4 is about 30-40 minutes, and the total time required to generate a script using GPT3.5 is about 10-15 minutes. Among them, the latex generation of ppt takes 2-3 minutes, the script generation of GPT3.5 takes 1-2 minutes, the script generation of GPT4 takes 15-20 minutes, and the voice generation of a 5-6 minute video takes 1-2 minutes. Video synthesis and processing are greatly affected by computer performance and video length, and it is roughly estimated to be about 10-20 minutes. In terms of cost, if GPT4 is used throughout the process to pursue quality, the final video of 16-17 minutes will cost 1.1-1.2 dollars. If GPT3.5 is used for script generation, the video length will be shortened to 5-6 minutes, and the cost will drop to 40-50 cents. If the image generation link is removed, the cost will drop to 30-35 cents. If the voice generation link is removed, the cost will drop to 10-20 cents (mainly from GPT generating slides).
