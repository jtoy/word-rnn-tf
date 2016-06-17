require 'yaml'
modelfile = YAML.load_file(ARGV[0])

steps =  4
permutations = []
params = modelfile['Tuning']['optimization']
params.each do |key,v|
  current_step = min = v['min']
  max = v['max']
  step = (max - min) / steps
  step = 1 if step == 0

  temp = []
  while current_step <= max
    temp << "#{key} #{current_step}"
    current_step += step
  end

  if permutations.empty?
    permutations = temp
  else
    permutations = permutations.product temp
  end
end

default_params = []
modelfile['Tuning']['default'].each do |k, v|
  default_params << "#{k} #{v}"
end

permutations.each do |x|
  puts (x + default_params).join(' ')
end


